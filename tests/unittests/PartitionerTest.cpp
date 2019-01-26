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
#include "glow/ExecutionEngine/ExecutionEngine.h"
#include "glow/Graph/Graph.h"

#include "gtest/gtest.h"

using namespace glow;

class PartitionerTest : public ::testing::Test {
public:
  PartitionerTest() : F_(mod_.createFunction("main")) {}

protected:
  Module mod_;
  Function *F_;
  Context ctx_;
};

/// Execute a graph of functions based on the given DAG.
static void executeDAG(DAGNode *G, Module &mod, Context &ctx,
                       llvm::ArrayRef<Placeholder *> vars,
                       llvm::ArrayRef<Tensor *> inputs) {
  std::unordered_map<std::string, Function *> name2func;

  for (auto *F : mod.getFunctions()) {
    name2func[F->getName()] = F;
  }

  std::vector<DAGNode *> exeList;
  int endPt = 0;
  int curPt = 0;
  // The first node is always the dummy node.
  exeList.push_back(G);
  endPt++;
  while (curPt < endPt) {
    DAGNode *dag = exeList.at(curPt);
    // The root in a G is always a dummy function.
    if (curPt > 0) {
      ExecutionEngine EE;
      Function *func = name2func[dag->name];
      EE.compile(CompilationMode::Infer, func);
      updateInputPlaceholders(ctx, vars, inputs);
      EE.run(ctx);
    }
    for (int i = 0, e = dag->children.size(); i < e; i++) {
      exeList.push_back(dag->children.at(i));
      endPt++;
    }
    curPt++;
  }
}

/// This one tests the model with this feature: after BFS, the memory
/// comsumption of all the nodes in each level won't exceed the device memory
/// constraints.
TEST_F(PartitionerTest, Basic1) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 32}, "input", false);
  auto *w1 = mod_.createConstant(ElemKind::FloatTy, {32, 16}, "w1");
  auto *b1 = mod_.createConstant(ElemKind::FloatTy, {16}, "b1");
  ctx_.allocate(input);
  w1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b1->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());

  // Initial FC.
  Node *I = F_->createFullyConnected("initial_fc", input, w1, b1);
  I = F_->createSigmoid("initial_sigmoid", I);

  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", I, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", I, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *ctx_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 32});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(ctx_, {input}, {&in});
  EE.run(ctx_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices = {{3072}, {3072}, {3072}};
  Partitioner myPartitioner(&mod_, devices);

  DAGNodeList myList = std::move(myPartitioner.Partition());
  ASSERT_EQ(mod_.getFunctions().size(), 3);
  ASSERT_EQ(myList.roots.size(), 1);

  // Run the paritioned graph and compare the results.
  ctx_.allocate(mod_.getPlaceholders());
  for (auto it = myList.roots.begin(); it != myList.roots.end(); ++it) {
    ctx_.allocate(mod_.getPlaceholders());
    executeDAG((*it).get(), mod_, ctx_, {input}, {&in});
    Tensor test = res.clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}

/// This one tests the model with this feature: after BFS, there is one level,
/// the  memory comsumption of all the nodes in which exceeds the device memory
/// constraints.
TEST_F(PartitionerTest, Basic2) {
  auto *input =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input", false);
  auto *input1 =
      mod_.createPlaceholder(ElemKind::FloatTy, {1, 16}, "input1", false);
  ctx_.allocate(input);
  ctx_.allocate(input1);
  // Left branch.
  auto *w2 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w2");
  auto *b2 = mod_.createConstant(ElemKind::FloatTy, {16}, "b2");
  w2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b2->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *L = F_->createFullyConnected("left_fc1", input, w2, b2);
  L = F_->createSigmoid("left_sigmoid1", L);
  auto *w3 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w3");
  auto *b3 = mod_.createConstant(ElemKind::FloatTy, {8}, "b3");
  w3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b3->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  L = F_->createFullyConnected("left_fc2", L, w3, b3);
  L = F_->createSigmoid("left_sigmoid2", L);

  // Right branch.
  auto *w4 = mod_.createConstant(ElemKind::FloatTy, {16, 16}, "w4");
  auto *b4 = mod_.createConstant(ElemKind::FloatTy, {16}, "b4");
  w4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b4->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  Node *R = F_->createFullyConnected("right_fc1", input1, w4, b4);
  R = F_->createSigmoid("right_sigmoid1", R);
  auto *w5 = mod_.createConstant(ElemKind::FloatTy, {16, 8}, "w5");
  auto *b5 = mod_.createConstant(ElemKind::FloatTy, {8}, "b5");
  w5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  b5->getHandle<>().randomize(-2.0, 2.0, mod_.getPRNG());
  R = F_->createFullyConnected("right_fc2", R, w5, b5);
  R = F_->createSigmoid("right_sigmoid2", R);

  // Join branches.
  auto *mul = F_->createMul("mul", L, R);
  auto *save = F_->createSave("ret", mul);
  auto &res = *ctx_.allocate(save->getPlaceholder());

  // Infer using the un-partitioned graph.
  Tensor in(ElemKind::FloatTy, {1, 16});
  ExecutionEngine EE;

  EE.compile(CompilationMode::Infer, F_);
  updateInputPlaceholders(ctx_, {input, input1}, {&in, &in});
  EE.run(ctx_);
  Tensor ref = res.clone();

  std::vector<DeviceInfo> devices = {{2048}, {2048}, {2048}};
  Partitioner myPartitioner(&mod_, devices);

  DAGNodeList myList = std::move(myPartitioner.Partition());
  //ASSERT_EQ(mod_.getFunctions().size(), 3);
  ASSERT_EQ(myList.roots.size(), 1);

  // Run the paritioned graph and compare the results.
  ctx_.allocate(mod_.getPlaceholders());
  for (auto it = myList.roots.begin(); it != myList.roots.end(); ++it) {
    ctx_.allocate(mod_.getPlaceholders());
    executeDAG((*it).get(), mod_, ctx_, {input}, {&in});
    Tensor test = res.clone();
    EXPECT_TRUE(ref.isEqual(test));
  }
}
