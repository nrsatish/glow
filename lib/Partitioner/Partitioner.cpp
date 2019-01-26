/**
 * Copyright (c) 2017-present, Facebook, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "glow/Partitioner/Partitioner.h"
#include "glow/Graph/Context.h"
#include "glow/Graph/Utils.h"

#include "llvm/Support/Casting.h"
#include "llvm/Support/raw_ostream.h"

using namespace glow;
using llvm::isa;

/// A graph with BFS oder.
struct BFSLevel {
  /// A list of <level, nodelist> with BFS order.
  std::vector<std::pair<int, std::vector<Node *>>> levels;
  /// A set of visited nodes.
  std::unordered_set<const Node *> visited;
};

Partitioner::Partitioner(Module *parent, const std::vector<DeviceInfo> &devices)
    : module_(parent), deviceInfo_(devices) {
  memSize_ = module_->getConstantsSize();
}

Function *Partitioner::selectRepFunc(Module *parent, size_t &memSize) {
  auto funcList = parent->getFunctions();
  Function *ret = nullptr;
  for (Function *F : funcList) {
    size_t size = memSize;

    // The set to keep the placeholders (only for Inputs) whose size is
    // already calculated.
    std::set<llvm::StringRef> pSet;

    for (auto &node : F->getNodes()) {
      int n = node.getNumInputs();
      if (node.getKind() == Kinded::Kind::SaveNodeKind) {
        // Special node, the placeholder should be ignored?
        continue;
      }
      for (int i = 0; i < n; i++) {
        Placeholder *in =
            llvm::dyn_cast<Placeholder>(node.getNthInput(i).getNode());
        if (in && pSet.count(in->getName()) == 0) {
          auto ty = in->getType();
          size += ty->getSizeInBytes();
          pSet.insert(in->getName());
        }
      }
    }
    // Find the function with largest required memory as the representive
    // function.
    if (size > memSize) {
      ret = F;
      memSize = size;
    }
  }
  return ret;
}

/// Get the minimal memory requirement (constant) for each op in the function.
void Partitioner::initOpMemUsage() {
  memUsage_.clear();
  for (auto &node : F_->getNodes()) {
    int n = node.getNumInputs();
    unsigned size = 0;
    if (node.getKind() == Kinded::Kind::SaveNodeKind) {
      memUsage_[&node] = size;
      continue;
    }
    for (int i = 0; i < n; i++) {
      Storage *in = llvm::dyn_cast<Storage>(node.getNthInput(i).getNode());
      if (in) {
        auto ty = in->getType();
        size += ty->getSizeInBytes();
      }
    }
    memUsage_[&node] = size;
  }
}

/// Get the minimal compute time for each op in the function.
void Partitioner::initOpComputeTime() {
  computeTime_.clear();

  // float peak_flops_fp16 = 10e+12; // assuming 10 TOPS/sec per card in FP16
  float peak_dram_bw = 60e+9; // assuming 60 GBytes/second from DDR per card
  float peak_sram_bw = 1e+12; // assuming 1 TB/second from SRAM per card

  for (auto &node : F_->getNodes()) {
    /// compute a simple roofline for now.
    int n = node.getNumInputs();
    uint64_t size_dram = 0;
    uint64_t size_sram = 0;
    if (node.getKind() == Kinded::Kind::SaveNodeKind) {
      computeTime_[&node] = std::max(size_dram * 1.0 / peak_dram_bw,
                                     size_sram * 1.0 / peak_sram_bw);

      continue;
    }

    for (int i = 0; i < n; i++) {
      if ( (node.getKind() == Kinded::Kind::SparseLengthsWeightedSumNodeKind ||
            node.getKind() == Kinded::Kind::RowwiseQuantizedSparseLengthsWeightedSumNodeKind)
            && llvm::isa<Storage>(node.getNthInput(i).getNode()) ) {
        auto ty = node.getNthInput(i).getNode()->getType(0);
        size_dram += ty->getSizeInBytes();
      }
      else {
        auto ty = node.getNthInput(i).getNode()->getType(0);
        size_sram += ty->getSizeInBytes();
      }
    }
    computeTime_[&node] = std::max(size_dram * 1.0f / peak_dram_bw,
                                   size_sram * 1.0f / peak_sram_bw);

  }
}

static BFSLevel getBFSLevel(Function *F) {
  // Visit graph nodes in BFS order. For each non-storage node, get its level.
  // Use the preorder to get the roots. Now assume there is only one output op
  // (i.e. root) now.
  GraphPreOrderVisitor visitor(*F);
  Node *node = nullptr;
  for (auto &N : visitor.getPreOrder()) {
    if (isa<Storage>(N)) {
      continue;
    }
    node = N;
    break;
  }

  BFSLevel bfs;
  int level = 0;
  int current = 0;
  bfs.levels.push_back({level, {node}});
  bfs.visited.insert(node);
  level++;
  while (current < level) {
    std::vector<Node *> nodes;
    for (int i = 0, e = bfs.levels[current].second.size(); i < e; i++) {
      Node *N = bfs.levels[current].second[i];

      for (int j = 0, e = N->getNumInputs(); j < e; ++j) {
        Node *in = N->getNthInput(j).getNode();
        if (isa<Storage>(in) || bfs.visited.count(in)) {
          continue;
        }
        nodes.push_back(in);
        bfs.visited.insert(in);
      }
    }
    if (nodes.size() > 0) {
      auto newPair = std::make_pair(level, nodes);
      bfs.levels.push_back(newPair);
      level++;
    }
    current++;
  }

  return bfs;
}

/// Assign nodes to partitions and return the mapping.
NodeToFunctionMap Partitioner::selectPartitions(Function *F,
                                                unsigned availableMemory) {
  NodeToFunctionMap mapping;
  BFSLevel bfs = getBFSLevel(F);
  unsigned level = bfs.levels.size();
  // A list of cut. The graph can be partitioned by levels [level - 1,
  // cut[0]), [cut[0] - 1, cut[1]), ..., [cut[n], -1).
  std::vector<int> cut;

  // Step 1 : get the initial cut based on BFS levels and avaiableMemory.
  // TODO .. need to remove the duplicated memory usage.
  unsigned mem = 0;
  for (int i = level - 1; i >= 0; i--) {
    unsigned tmp = 0;
    for (int j = 0, e = bfs.levels[i].second.size(); j < e; j++) {
      Node *N = bfs.levels[i].second[j];
      tmp += memUsage_[N];
    }
    if (mem + tmp > availableMemory) {
      if (mem == 0) {
        // This means the mem usage for one level exceeds the availableMem,
        // accept it now and will do adjustment later.
        cut.push_back(i + 1);
      } else {
        cut.push_back(i);
        mem = tmp;
      }
    } else {
      mem += tmp;
    }
  }

  // The last border.
  cut.push_back(-1);

  // Step 2 : Create the initial mapping between node and functions.
  for (int k = 0, e = cut.size(); k < e; k++) {
    auto *newF = F->getParent()->createFunction(std::string(F->getName()) +
                                                "_part" + std::to_string(k));
    mapping.createPartition(newF);
    for (int i = k > 0 ? cut[k - 1] : level - 1; i > cut[k]; i--) {
      for (int j = 0, e1 = bfs.levels[i].second.size(); j < e1; j++) {
        Node *N = bfs.levels[i].second[j];
        mapping.add(N, newF);
      }
    }
  }
  // Step 3 : adjust the partition based on performance (Advanced Graph
  // Paritioning algrithm will be applied here).
  // --- TODO

  return mapping;
}

/// Create a NodeToFunctionMap structure from an assignment
void Partitioner::generate_mapping(Function *F, std::unordered_map<Node*, int>& assignment, NodeToFunctionMap& mapping) {
  /// This combines connected subgraphs with the same assignment to a function

  /// Essentially traverses one connected component at a time using BFS
  BFSLevel bfs;
  int k = 0; // this is the partition id

  while(true) {
    // find a node that hasnt been "visited" i.e. assigned to a partition
    GraphPreOrderVisitor visitor(*F);
    Node *node = nullptr;

    for (auto &N : visitor.getPreOrder()) {
      if (isa<Storage>(N) || bfs.visited.count(N)) {
        continue;
      }
      node = N;
      break;
    }
    if (node == nullptr) {
      // every non-storage node has been assigned to a partition; we are done
      break;
    }

    // we have a root starting node for BFS
    // we want to find all connected nodes that have the same processor assignment
    // These get assigned to a partition.

    // TODO: Fix this so that multiple root nodes at the same level can get
    // assigned to the same partition.
    bfs.levels.clear();
    int proc_id = assignment[node];
    int current = 0;
    int level = 0;
    bfs.levels.push_back({level, {node}});
    bfs.visited.insert(node);
    auto *newF = F->getParent()->createFunction(std::string(F->getName()) +
                                                "_2_part" + std::to_string(k));

    mapping.createPartition(newF);
    // printf("Created new partition: %d for proc_id: %d\n", k, proc_id);
    mapping.add(node, newF);
    // printf("Added node: %s to partition %d\n", node->getName().str().c_str(), k);

    level++;
    while (current < level) {
      std::vector<Node *> nodes;
      for (int i = 0, e = bfs.levels[current].second.size(); i < e; i++) {
        Node *N = bfs.levels[current].second[i];

        for (int j = 0, e = N->getNumInputs(); j < e; ++j) {
          Node *in = N->getNthInput(j).getNode();
          if (isa<Storage>(in) || bfs.visited.count(in) || assignment[in] != proc_id) {
            continue;
          }
          nodes.push_back(in);
          mapping.add(in, newF);
          bfs.visited.insert(in);
          //printf("Added node: %s to partition %d\n", in->getName().str().c_str(), k);
        }
      }
      if (nodes.size() > 0) {
        auto newPair = std::make_pair(level, nodes);
        bfs.levels.push_back(newPair);
        level++;
      }
      current++;
    }
    // finished current partition, increment partition id and find an unvisited node
    k++;
  }

  return;
}

// Storage of costs for processors and links.
struct ProcessorCost {
  std::vector<float> egress_costs;
  std::vector<float> ingress_costs;
  std::vector<float> processor_costs;
  std::vector<unsigned> processor_memory_available;
};

/// Compute finish time given (1) an (partial) allocation of nodes to processors
/// and (2) a tentative assignment of a node "node" to a processor "tentative_assignment"
static float computeAllocationCost(std::unordered_map<Node*, int>& assignment,
                                         struct ProcessorCost& current_cost,
                                         Node* node,
                                         int tentative_assignment,
                                         float node_cost,
                                         struct ProcessorCost& tentative_cost) {
  // The idea here is to incrementally update current_cost
  // (1) Add node_cost to processor_costs to reflect the assignment
  tentative_cost = current_cost;
  tentative_cost.processor_costs[tentative_assignment] += node_cost;

  uint64_t my_comm_size = 0;
  if(node->getNumResults() > 0) {
    auto myty = node->getType(0);
    my_comm_size = myty->getSizeInBytes();
  }

  // (2) Find all inputs that are not assigned to the same node and add
  //     the communication cost to egress of input and ingress of current node

  for (int j = 0, e = node->getNumInputs(); j < e; ++j) {
    Node *in = node->getNthInput(j).getNode();
    if (isa<Storage>(in)) {
      continue;
    }
    auto ty = in->getType(0);
    uint64_t comm_size = ty->getSizeInBytes();
    if ( assignment.find(in) != assignment.end() && assignment[in] != tentative_assignment) {
      tentative_cost.egress_costs[assignment[in]] += comm_size / 3.2e9f;
      tentative_cost.ingress_costs[tentative_assignment] += comm_size / 3.2e9f;
      //printf("Requires egress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, assignment[in], tentative_cost.egress_costs[assignment[in]]);
      //printf("Requires ingress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, tentative_assignment, tentative_cost.ingress_costs[tentative_assignment]);
    }
  }
  // (3) Find all outputs that are not assigned to the same node and add
  //     the communication cost to ingress of input and egress of current node
  // for outputs, comm size is own produced size.
  for (int j = 0, e = node->getNumResults(); j < e; ++j){
    Node *out = node->getNthResult(j).getNode();
    for (auto it = out->getUsers().begin(); it != out->getUsers().end(); ++it) {
      Node* out = it->getUser();
      //printf("User: %s\n", out->getName().str().c_str());

      if ( assignment.find(out) != assignment.end() && assignment[out] != tentative_assignment) {
        tentative_cost.ingress_costs[assignment[out]] += my_comm_size / 3.2e9f;
        tentative_cost.egress_costs[tentative_assignment] += my_comm_size / 3.2e9f;
        //printf("Requires ingress cost of %e for proc %d (cur cost = %e)\n", my_comm_size / 3.2e9, assignment[out], tentative_cost.ingress_costs[assignment[out]]);
        //printf("Requires egress cost of %e for proc %d (cur cost = %e)\n", my_comm_size / 3.2e9, tentative_assignment, tentative_cost.egress_costs[tentative_assignment]);
      }
    }
  }

  // Then find the maximum cost of any of the components to find overall bottleneck
  float cost = 0;
  for (float e : tentative_cost.egress_costs) {
    cost = std::max(cost, e);
  }
  for (float i : tentative_cost.ingress_costs) {
    cost = std::max(cost, i);
  }
  for (float p : tentative_cost.processor_costs) {
    cost = std::max(cost, p);
  }
  return cost;
}

/// Assign nodes to partitions and return the mapping.
NodeToFunctionMap Partitioner::selectPartitions2(Function *F,
                                                unsigned availableMemory,
                                                unsigned num_processors) {
  NodeToFunctionMap mapping;

  // The following is a copy of selectPartitions for comparison sake.
  /*
  ProcessorCost cost;
  for (int i = 0; i < num_processors; i++) {
    cost.processor_costs.push_back(0.f);
    cost.processor_memory_available.push_back(availableMemory);
  }
  BFSLevel bfs = getBFSLevel(F);
  unsigned level = bfs.levels.size();
  // A list of cut. The graph can be partitioned by levels [level - 1,
  // cut[0]), [cut[0] - 1, cut[1]), ..., [cut[n], -1).
  std::vector<int> cut;

  // Step 1 : get the initial cut based on BFS levels and avaiableMemory.
  // TODO .. need to remove the duplicated memory usage.
  unsigned mem = 0;
  for (int i = level - 1; i >= 0; i--) {
    unsigned tmp = 0;
    for (int j = 0, e = bfs.levels[i].second.size(); j < e; j++) {
      Node *N = bfs.levels[i].second[j];
      tmp += memUsage_[N];
    }
    if (mem + tmp > availableMemory) {
      if (mem == 0) {
        // This means the mem usage for one level exceeds the availableMem,
        // accept it now and will do adjustment later.
        cut.push_back(i + 1);
      } else {
        cut.push_back(i);
        mem = tmp;
      }
    } else {
      mem += tmp;
    }
  }

  // The last border.
  cut.push_back(-1);

  // Step 2 : Create the initial mapping between node and functions.
  for (int k = 0, e = cut.size(); k < e; k++) {
    auto *newF = F->getParent()->createFunction(std::string(F->getName()) +
                                                "_part" + std::to_string(k));
    mapping.createPartition(newF);
    for (int i = k > 0 ? cut[k - 1] : level - 1; i > cut[k]; i--) {
      for (int j = 0, e1 = bfs.levels[i].second.size(); j < e1; j++) {
        Node *N = bfs.levels[i].second[j];
        mapping.add(N, newF);
        cost.processor_memory_available[k] -= memUsage_[N];
        cost.processor_costs[k] += computeTime_[N];
      }
    }
  }

  // Print out actual costs and memory
  for (int i = 0; i < num_processors; i++) {
    printf("BFS ::: Proc: %d cost: %e memory_available: %d\n", i, cost.processor_costs[i], cost.processor_memory_available[i]);
  }
  */

  /// Cost based scheduling
  /// This is based on a first fit decreasing bib packing
  /// We define bins for each processor and each communication channel
  /// 1. Sort all nodes on the basis of decrasing cost
  /// 2. Assign each node to the processor on which it has the earlier finish time and can fit in memory
  ///    Take into account communication from any allocations of input/output nodes
  /// 3. If at some point we cannot find any processor avaialble due to memory fragmentation,
  ///    restart step 1 with a sort based on decreasing memory size.
  NodeToFunctionMap mapping2;

  // We first try to sort on compute
  bool sortoncompute = true;
  bool sortonmemory = false;

  // Loop that will eventually try both options and break out if both dont work.
  bool failed = false;
  while(sortoncompute || sortonmemory) {
    // Keep track of processor costs, memory available, ingress BW and egress BW
    ProcessorCost cost;
    cost.processor_costs.resize(num_processors, 0.f);
    cost.processor_memory_available.resize(num_processors, availableMemory);
    cost.egress_costs.resize(num_processors, 0.f);
    cost.ingress_costs.resize(num_processors, 0.f);

    // Make a copy of Node pointers since we need to sort the nodes below
    unsigned nnodes = F->getNodes().size();
    std::vector<Node *> nodes;
    for (auto &node : F->getNodes()) {
      nodes.push_back(&node);
    }

    // Sort the node pointers based on memory or compute
    if (sortoncompute) {
      ComputeTimeMap& computetime = computeTime_;
      std::sort(nodes.begin(), nodes.end(), [&computetime](Node* n1, Node* n2) {
        return computetime[n1] > computetime[n2];
      });
    } else if (sortonmemory) {
      MemUsageMap& memusage = memUsage_;
      std::sort(nodes.begin(), nodes.end(), [&memusage](Node* n1, Node* n2) {
        return memusage[n1] > memusage[n2];
      });
    } else {
      failed = true;
      break;
    }

    // We iterate through the nodes in sorted order.
    // For each node, we try to assign it greedily to a processor
    // where the overall finish time is lowest. This takes into account
    // how busy the processor is and communication costs between the
    // node and its predecessors and successors.
    bool failed_inner = false;
    std::unordered_map<Node*, int> assignment(nnodes);
    for (auto &node : nodes) {
      //printf("Considering node: %s cost: %e memory: %d\n", node->getName().str().c_str(), computeTime_[node], memUsage_[node]);

      // Greedy choice of processor. Try all processors, and for each tentative
      // assignment of the node to each processor, calculate estimated finish time.
      // Choose minimum over all esimated finish times.
      int min_cost_proc = -1;
      float min_cost = 1e+10;

      for (int i = 0; i < num_processors; i++) {
        //printf("Proc: %d cost: %e memory_available: %d\n", i, cost.processor_costs[i], cost.processor_memory_available[i]);
        ProcessorCost tentative_cost;
        float estimated_cost = computeAllocationCost(assignment, cost, node, i, computeTime_[node], tentative_cost);

        //printf("Effective finish time: %e\n", estimated_cost);
        if (estimated_cost < min_cost
            && cost.processor_memory_available[i] >= memUsage_[node]) {
          min_cost = estimated_cost;
          min_cost_proc = i;
        }
      }

      if (min_cost_proc == -1) {
        // we didnt find any valid allocation.
        failed_inner = true;
        break;
      }

      assignment[node] = min_cost_proc;
      cost.processor_memory_available[min_cost_proc] -= memUsage_[node];
      ProcessorCost next_cost;
      computeAllocationCost(assignment, cost, node, min_cost_proc, computeTime_[node], next_cost);
      cost.processor_costs = next_cost.processor_costs;
      cost.ingress_costs = next_cost.ingress_costs;
      cost.egress_costs = next_cost.egress_costs;

    }

    if (failed_inner) {
      if (sortoncompute) {
        sortoncompute = false;
        sortonmemory = true;
        continue;
      }
      else {
        // we have tried both, bail out.
        failed = true;
        break;
      }
    }
    /*
    for (auto &node : nodes) {
      printf("Node: %s assignment: %d\n", node->getName().str().c_str(), assignment[node]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc: %d cost: %e memory_available: %d\n", i, cost.processor_costs[i], cost.processor_memory_available[i]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc egress: %d cost: %e \n", i, cost.egress_costs[i]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc ingress: %d cost: %e \n", i, cost.ingress_costs[i]);
    }
    */
    generate_mapping(F, assignment, mapping2);
    break;
  }

  return mapping2;
}

/// Adjust the logicalDevice ID for each DAGNode. This happens when \p num (i.e.
/// the number of DAGNodes) is larger than the number of devices. E.g:
/// node1(6GB) -> node2(14GB) -> node3(6GB). The memory limitation is 16GB, and
/// there is only 2 devices.
void Partitioner::adjustLogicalDeviceID(DAGNode *DAG, int num) {}

/// Current only partition the representive function.
void Partitioner::doPartitioning(Function *F, NodeToFunctionMap &mapping) {
  // The dummy node.
  std::unique_ptr<DAGNode> DAG = std::make_unique<DAGNode>();
  DAG->logicalDevice = 0;
  DAG->name = F->getName();
  DAG->deviceID = 0;
  DAG->logicalDevice = 0;
  DAGNode *root = DAG.get();
  partitions_.roots.push_back(std::move(DAG));
  llvm::DenseMap<Node *, Node *> currToNew;

  // Clone nodes into target partition.
  for (auto &N : F->getNodes()) {
    auto *clone = N.clone();
    currToNew[&N] = clone;
    mapping[&N]->addNode(clone);
  }

  // For any dependency that crosses a partition, add a placeholder and save
  // node. Record the dependence in the function graph.
  int logicalID = 0;
  llvm::DenseMap<Node *, Placeholder *> placeholders;
  llvm::DenseMap<Function *, DAGNode *> funcDAG;
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG.find(subF) == funcDAG.end()) {
      std::unique_ptr<DAGNode> subDAG = std::make_unique<DAGNode>();
      subDAG->name = subF->getName();
      subDAG->logicalDevice = logicalID++;
      funcDAG[subF] = subDAG.get();
      partitions_.nodes.push_back(std::move(subDAG));
    }

    // Link subF to its parents.
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        if (isa<Storage>(input.getNode()))
          continue;

        auto *inputF = mapping[input.getNode()];
        if (subF == inputF)
          continue;

        // Check if a DAGNode for subF's parent is created or not. If not,
        // create one.
        if (funcDAG.find(inputF) == funcDAG.end()) {
          std::unique_ptr<DAGNode> subDAG = std::make_unique<DAGNode>();
          subDAG->name = inputF->getName();
          subDAG->logicalDevice = logicalID++;
          funcDAG[inputF] = subDAG.get();
          partitions_.nodes.push_back(std::move(subDAG));
        }

        // subF is a child of inputF, inputF is a parent of subF.
        funcDAG[inputF]->children.push_back(funcDAG[subF]);
        funcDAG[subF]->parents.push_back(funcDAG[inputF]);

        // If we've already created a placeholder for this dependence, use it.
        auto it = placeholders.find(input.getNode());
        if (it != placeholders.end()) {
          N.setNthInput(inp, it->second);
          continue;
        }

        // Create a new placeholder to represent this dependence.
        auto *save = inputF->createSave("tmp", input);
        auto *tmp = save->getPlaceholder();
        placeholders[input.getNode()] = tmp;
        N.setNthInput(inp, tmp);
      }
    }
  }

  // Update links between nodes in the cloned functions. Add placeholders (and
  // save nodes) where a link crosses a partition boundary.
  for (auto *subF : mapping.getPartitions()) {
    for (auto &N : subF->getNodes()) {
      for (int inp = 0, e = N.getNumInputs(); inp < e; inp++) {
        auto input = N.getNthInput(inp);
        if (isa<Storage>(input.getNode()))
          continue;
        // Link this node to the clone of its input.
        auto *clone = currToNew[input.getNode()];
        N.setNthInput(inp, NodeValue(clone, input.getResNo()));
      }
    }
  }

  // For all DAGNode without parents, link them to the root DAG.
  for (auto *subF : mapping.getPartitions()) {
    if (funcDAG[subF]->parents.size() == 0) {
      funcDAG[subF]->parents.push_back(DAG.get());
      root->children.push_back(funcDAG[subF]);
    }
  }

  // Adjust the logicalDevice for each DAGNode.
  if (mapping.getPartitions().size() > deviceInfo_.size()) {
    adjustLogicalDeviceID(DAG.get(), mapping.getPartitions().size());
  }
}

DAGNodeList &Partitioner::Partition() {

  // Find the representive function for running partitioning algrithm.
  F_ = selectRepFunc(module_, memSize_);

  // print out dag before partition1
  F_->dumpDAG("before");

  // Possible minimal k devices for a succesful partitioning
  // Note: here 2 is for testing;
  unsigned k = 2; //(memSize_ + MARGIN) / devices[0].availableMemory;

  if (k == 1) {
    // No partition is needed. Create DAGNode and return. This root is alway a
    // dummy function.
    for (auto F : module_->getFunctions()) {
      std::unique_ptr<DAGNode> DAG = std::make_unique<DAGNode>();
      DAG->logicalDevice = 0;
      DAG->name = F->getName();
      std::unique_ptr<DAGNode> DAG1 = std::make_unique<DAGNode>();
      DAG1->logicalDevice = 0;
      DAG1->name = F->getName();
      DAG1->parents.push_back(DAG.get());
      DAG->children.push_back(DAG1.get());
      partitions_.roots.push_back(std::move(DAG));
      partitions_.nodes.push_back(std::move(DAG1));
    }
    return partitions_;
  }

  // Prepare 1: Get the min memory usage for each op.
  initOpMemUsage();

  // Prepare 2: Get the roofline memory bandwidth estimate for each op.
  initOpComputeTime();

  // Prepare 3: TODO: get the minimal comunication cost for any 2 ops (i.e. the
  // output data size) Will calculate it on the fly. -- Will double check which
  // way is better.

  // Partition
  // Use BFS to do the initial partitioning. Starting from the final node, BFS
  // until the memory limitation reached one by one.
  unsigned unitMem = memSize_ / k * 1.2; // used for testing

  //NodeToFunctionMap partitionMap = selectPartitions(F_, unitMem);

  // Use bin packing method to find partitioning.
  NodeToFunctionMap partitionMap = selectPartitions2(F_, unitMem, k);

  doPartitioning(F_, partitionMap);

  // Remove the original function after partitioning.
  module_->eraseFunction(F_);

  auto funcList = module_->getFunctions();
  for (Function *F : funcList) {
    (void)F;
    assert(F->verify() && "Conversion led to invalid function");
  }

  int count = 0;
  for (Function *F : funcList) {
    F->dumpDAG("after_" + std::to_string(count));
    count++;
  }
  // TODO: Optional: if (k < number of devices)
  // Check the computation time of each sub-module, and find out the "key"
  // sub-module to decide if duplicating the sub-module is necessary.

  return partitions_;
}
