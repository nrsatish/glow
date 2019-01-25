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
      // every non-storage node has been assigned to a function; we are done
      break;
    }

    // we have a starting node for BFS
    bfs.levels.clear();
    int proc_id = assignment[node];
    int current = 0;
    int level = 0;
    bfs.levels.push_back({level, {node}});
    bfs.visited.insert(node);
    auto *newF = F->getParent()->createFunction(std::string(F->getName()) +
                                                "_2_part" + std::to_string(k));

    mapping.createPartition(newF);
    printf("Created new partition: %d for proc_id: %d\n", k, proc_id);
    mapping.add(node, newF);
    printf("Added node: %s to partition %d\n", node->getName().str().c_str(), k);

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
          printf("Added node: %s to partition %d\n", in->getName().str().c_str(), k);
        }
      }
      if (nodes.size() > 0) {
        auto newPair = std::make_pair(level, nodes);
        bfs.levels.push_back(newPair);
        level++;
      }
      current++;
    }
    k++;
  }

  return;
}

/// Assign nodes to partitions and return the mapping.
NodeToFunctionMap Partitioner::selectPartitions2(Function *F,
                                                unsigned availableMemory,
                                                unsigned num_processors) {
  NodeToFunctionMap mapping;
  std::vector<float> processor_costs(num_processors);
  std::vector<unsigned> processor_memory_available(num_processors);
  for (int i = 0; i < num_processors; i++) {
    processor_costs[i] = 0.f;
    processor_memory_available[i] = availableMemory;
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
        processor_memory_available[k] -= memUsage_[N];
        processor_costs[k] += computeTime_[N];
      }
    }
  }
  for (int i = 0; i < num_processors; i++) {
    printf("BFS ::: Proc: %d cost: %e memory_available: %d\n", i, processor_costs[i], processor_memory_available[i]);
  }


  NodeToFunctionMap mapping2;

  while(true) {
    int comm_devices = 2*num_processors; // ingress and egress for each proc
    std::vector<float> processor_costs(num_processors + comm_devices);
    std::vector<unsigned> processor_memory_available(num_processors);

    for (int i = 0; i < (num_processors + comm_devices); i++) {
      processor_costs[i] = 0.f;
    }
    for (int i = 0; i < num_processors; i++) {
      processor_memory_available[i] = availableMemory;
    }

    bool sortoncompute = true;

    // First we sort nodes based on decreasing costs.
    unsigned nnodes = F->getNodes().size();
    std::vector<Node *> nodes;
    for (auto &node : F->getNodes()) {
      nodes.push_back(&node);
    }


    if (sortoncompute) {
      ComputeTimeMap& computetime = computeTime_;
      std::sort(nodes.begin(), nodes.end(), [&computetime](Node* n1, Node* n2) {
        return computetime[n1] > computetime[n2];
      });
    } else {
      MemUsageMap& memusage = memUsage_;
      std::sort(nodes.begin(), nodes.end(), [&memusage](Node* n1, Node* n2) {
        return memusage[n1] > memusage[n2];
      });
    }

    // We iterate through the nodes in sorted order.
    // For each node, we try to assign it greedily to a processor. We have two
    // approaches possible here:
    // (1) we assign the node to the least busy (lowest cost) processor that can
    //     accommodate the node based on memory constraints
    // (2) we assign the node to the processor that has the most memory.
    // The first one will focus on load imbalance more than memory constraints;
    // the second is the reverse.

    // We likely want to try the first approach first. If it fails becasue at some
    // point we are unable to assign a node to any processor; then we fallback
    // to the second approach.
    std::unordered_map<Node*, int> assignment(nnodes);
    bool failed = false;
    for (auto &node : nodes) {
      printf("Considering node: %s cost: %e memory: %d\n", node->getName().str().c_str(), computeTime_[node], memUsage_[node]);
      int min_cost_proc = -1;
      float min_cost = 1e+10;
      for (int i = 0; i < num_processors; i++) {
        printf("Proc: %d cost: %e memory_available: %d\n", i, processor_costs[i], processor_memory_available[i]);

        std::vector<float> egress_costs(num_processors);
        std::vector<float> ingress_costs(num_processors);
        for(int j = 0; j < num_processors; j++) {
          egress_costs[j] = processor_costs[num_processors + j];
          ingress_costs[j] = processor_costs[2*num_processors + j];
        }

        for (int j = 0, e = node->getNumInputs(); j < e; ++j) {
          Node *in = node->getNthInput(j).getNode();
          if (isa<Storage>(in)) {
            continue;
          }
          auto ty = in->getType(0);
          uint64_t comm_size = ty->getSizeInBytes();
          if ( assignment.find(in) != assignment.end() && assignment[in] != i) {
            egress_costs[assignment[in]] += comm_size / 3.2e9f;
            ingress_costs[i] += comm_size / 3.2e9f;
            printf("Requires egress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, assignment[in], processor_costs[num_processors + assignment[in]]);
            printf("Requires ingress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, i, processor_costs[2*num_processors + i]);
          }
        }
        for (int j = 0, e = node->getNumResults(); j < e; ++j){
          Node *out = node->getNthResult(j).getNode();
          if (isa<Storage>(out)) {
            continue;
          }
          auto ty = out->getType(0);
          uint64_t comm_size = ty->getSizeInBytes();
          if ( assignment.find(out) != assignment.end() && assignment[out] != i) {
            ingress_costs[assignment[out]] += comm_size / 3.2e9f;
            egress_costs[i] += comm_size / 3.2e9f;
            printf("Requires ingress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, assignment[out], processor_costs[num_processors + assignment[out]]);
            printf("Requires egress cost of %e for proc %d (cur cost = %e)\n", comm_size / 3.2e9, i, processor_costs[2*num_processors + i]);
          }
        }

        float comm_cost = 0;
        for (float cost : egress_costs) {
          comm_cost = std::max(comm_cost, cost);
        }
        for (float cost : ingress_costs) {
          comm_cost = std::max(comm_cost, cost);
        }

        printf("Effective start time: %e\n", std::max(processor_costs[i], comm_cost));
        if (std::max(processor_costs[i], comm_cost) < min_cost
            && processor_memory_available[i] >= memUsage_[node]) {
          min_cost = std::max(processor_costs[i], comm_cost);
          min_cost_proc = i;
        }
      }
      if(min_cost_proc != -1) {
        assignment[node] = min_cost_proc;
        processor_memory_available[min_cost_proc] -= memUsage_[node];
        processor_costs[min_cost_proc] += computeTime_[node];
        for (int j = 0, e = node->getNumInputs(); j < e; ++j) {
          Node *in = node->getNthInput(j).getNode();
          if (isa<Storage>(in)) {
            continue;
          }
          auto ty = in->getType(0);
          uint64_t comm_size = ty->getSizeInBytes();
          if ( assignment.find(in) != assignment.end() && assignment[in] != min_cost_proc) {
            processor_costs[num_processors + assignment[in]] += comm_size / 3.2e9;
            processor_costs[2*num_processors + min_cost_proc] += comm_size / 3.2e9;
          }
        }
        for (int j = 0, e = node->getNumResults(); j < e; ++j){
          Node *out = node->getNthResult(j).getNode();
          if (isa<Storage>(out)) {
            continue;
          }
          auto ty = out->getType(0);
          uint64_t comm_size = ty->getSizeInBytes();
          if ( assignment.find(out) != assignment.end() && assignment[out] != min_cost_proc) {
            processor_costs[2*num_processors + assignment[out]] += comm_size / 3.2e9;
            processor_costs[num_processors + min_cost_proc] += comm_size / 3.2e9;
          }
        }
      } else {
        failed = true;
        break;
      }
    }

    if (failed) {
      sortoncompute = false;
      continue;
    }
    for (auto &node : nodes) {
      printf("Node: %s assignment: %d\n", node->getName().str().c_str(), assignment[node]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc: %d cost: %e memory_available: %d\n", i, processor_costs[i], processor_memory_available[i]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc egress: %d cost: %e \n", i, processor_costs[num_processors + i]);
    }
    for (int i = 0; i < num_processors; i++) {
      printf("Proc ingress: %d cost: %e \n", i, processor_costs[2*num_processors + i]);
    }
    generate_mapping(F, assignment, mapping2);
    break;
  }



  // Step 3 : adjust the partition based on performance (Advanced Graph
  // Paritioning algrithm will be applied here).
  // --- TODO

  return mapping;
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

  // TODO: Optional: if (k < number of devices)
  // Check the computation time of each sub-module, and find out the "key"
  // sub-module to decide if duplicating the sub-module is necessary.

  return partitions_;
}
