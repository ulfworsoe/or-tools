// Copyright 2026 Mosek
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Mosek backend to MPSolver.
//
#include <absl/base/attributes.h>
#include <strings.h>

#include <new>
#include <string_view>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <ranges>

//#include "absl/flags/flag.h"
#include "absl/log/check.h"
//#include "absl/log/die_if_null.h"
//#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/cleanup/cleanup.h"
//#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "ortools/base/timer.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver_callback.h"
//#include "ortools/linear_solver/proto_solver/proto_utils.h"
#include "ortools/third_party_solvers/mosekcore12_0_environment.h"
//#include "ortools/util/lazy_mutable_copy.h"
//#include "ortools/util/time_limit.h"

namespace operations_research {

namespace MSK = operations_research::MSK120;

class MosekInterface : public MPSolverInterface {
 public:
  explicit MosekInterface(MPSolver* solver, bool mip);
  ~MosekInterface() override;

  void SetOptimizationDirection(bool maximize) override;

  // ----- Solve -----
  // Solves the problem using the parameter values specified.
  MPSolver::ResultStatus Solve(const MPSolverParameters& param) override;

  // ----- Directly solve -----
  // Mosek should support being interrupted, but for now we'll only support
  // non-interupted solves.
  bool SupportsDirectlySolveProto(std::atomic<bool>* interrupt) const override {
    return false;
    // return interrupt == nullptr;
  }

  // Solve model dirctly, bypassing protobuffers
  // MPSolutionResponse DirectlySolveProto(LazyMutableCopy<MPModelRequest>
  // request,
  //                                      std::atomic<bool>* interrupt)
  //                                      override;

  // Writes the model.
  void Write(const std::string& filename) override;

  // ----- Model modifications and extraction -----
  // Resets extracted model
  void Reset() override;

  // Modifies bounds.
  void SetVariableBounds(int var_index, double lb, double ub) override;
  void SetVariableInteger(int var_index, bool integer) override;
  void SetConstraintBounds(int row_index, double lb, double ub) override;

  // Adds Constraint incrementally.
  void AddRowConstraint(MPConstraint* ct) override;
  bool AddIndicatorConstraint(MPConstraint* ct) override;
  // Adds variable incrementally.
  void AddVariable(MPVariable* var) override;
  // Changes a coefficient in a constraint.
  void SetCoefficient(MPConstraint* constraint, const MPVariable* variable,
                      double new_value, double old_value) override;
  // Clears a constraint from all its terms.
  void ClearConstraint(MPConstraint* constraint) override;
  // Changes a coefficient in the linear objective
  void SetObjectiveCoefficient(const MPVariable* variable,
                               double coefficient) override;
  // Changes the constant term in the linear objective.
  void SetObjectiveOffset(double value) override;
  // Clears the objective from all its terms.
  void ClearObjective() override;

  // ------ Query statistics on the solution and the solve ------
  // Number of simplex or interior-point iterations
  int64_t iterations() const override;
  // Number of branch-and-bound nodes. Only available for discrete problems.
  int64_t nodes() const override;

  // Returns the basis status of a row.
  MPSolver::BasisStatus row_status(int constraint_index) const override;
  // Returns the basis status of a column.
  MPSolver::BasisStatus column_status(int variable_index) const override;

  // ----- Misc -----
  // Queries problem type.
  bool IsLP() const override { return !mip_; }
  bool IsMIP() const override { return mip_; }
  bool IsContinuous() const override { return IsLP(); }

  void ExtractNewVariables() override;
  void ExtractNewConstraints() override;
  void ExtractObjective() override;

  int SolutionCount();
  std::string SolverVersion() const override {
    int major, minor, rev;

    MSK::get_version(&major, &minor, &rev);
    return absl::StrFormat("Mosek library version %d.%d.%d\n", major, minor,
                           rev);
  }

  bool InterruptSolve() override {
    break_solver_ = true;
    return true;
  }

  void* underlying_solver() override { return reinterpret_cast<void*>(task_); }

  double ComputeExactConditionNumber() const override {
    if (!IsContinuous()) {
      LOG(DFATAL) << "ComputeExactConditionNumber not implemented for"
                  << " MOSEK_MIXED_INTEGER_PROGRAMMING";
      return 0.0;
    }

    LOG(DFATAL) << "ComputeExactConditionNumber not implemented for"
                << " MOSEK_LINEAR_PROGRAMMING";
    return 0.0;
  }

  // Iterates through the solutions in Mosek's solution pool.
  bool NextSolution() override;

  void SetCallback(MPCallback* mp_callback) override;
  bool SupportsCallbacks() const override { return true; }

  void SetParameters(const MPSolverParameters&) override;
  bool SetSolverSpecificParametersAsString(
      const std::string& parameters) override;

  void SetRelativeMipGap(double value) override;
  void SetPrimalTolerance(double value) override;
  void SetDualTolerance(double value) override;
  void SetPresolveMode(int value) override;
  void SetScalingMode(int value) override;
  void SetLpAlgorithm(int value) override;

 private:
  void CheckedMosekCall(MSK::ResCode r) const;
  static void DeleteTask(MSK::Task_t task) { MSK::delete_task(task); }

  //----- Variables -----

  // The underlying MOSEK task, which is kept updated with all Add* calls.
  MSK::Task_t task_;
  bool break_solver_;
  std::unique_ptr<MSK::Task_s, void(*)(MSK::Task_t)> ptask_;

  bool mip_;
  // int current_solution_index_;
  //  mp callback function
  MPCallback* callback_ = nullptr;
  // Has length equal to the number of MPConstraints in
  // MPSolverInterface::solver_. Each entry is a triple (i,afei). If i is
  // non-negative, it maps to a ranged linear constraint, where i nad i+1 are
  // the indexes of the lower and upper bounds respectively, and afei is the
  // index of the row index. If i is negative is an indicator constraint, and
  // (-i-1) is the index of the DJC implementing the indicator constraint and
  // afei,afei+1 are the row indexes if the two rows required to define an
  // indicator constraint
  std::vector<std::tuple<int64_t,int64_t>> mp_cons_to_mosek_cons_;

  int64_t domidx_rfree;
  int64_t domidx_rzero;
  int64_t domidx_rplus;
  int64_t domidx_rminus;
};  // MosekInterface

namespace {

void MosekCloneParameters(MSK::Task_t tdst, MSK::Task_t tsrc) {

    //... TODO
  for (int p = MSK_DPAR_BEGIN; p < MSK_DPAR_END; ++p) {
    double parval;
    MSK_getdouparam(tsrc, (MSKdparame)p, &parval);
    MSK_putdouparam(tdst, (MSKdparame)p, parval);
  }
  for (int p = MSK_IPAR_BEGIN; p < MSK_IPAR_END; ++p) {
    int parval;
    MSK_getintparam(tsrc, (MSKiparame)p, &parval);
    MSK_putintparam(tdst, (MSKiparame)p, parval);
  }
}

std::pair<const char *, std::string> MosekLastError(MSK::Task_t task) {
    MSK::ResCode r = MSK::get_last_resp(task);
    const char * name = MSK::get_resp_name(r);
    auto len = MSK::get_last_resp_msg_len(task);
    std::vector<char> buf(len+1);
    MSK::get_last_resp_msg(task,buf.data(),buf.size());

    return { name,std::string(buf.data()) };
}

void CheckedMosekCall(MSK::Task_t task, MSK::ResCode r) {
  CHECK_EQ(MSK::RES_OK, r) << "Mosek Error " << MSK::get_resp_name(r) << ": "
                           << MosekLastError(task).second;
}

// This class provides a means of interacting with the task from the callback
// function.
class MosekMPCallbackContext : public MPCallbackContext {
  public:
  MosekMPCallbackContext();
  //MosekMPCallbackContext(MSK::Task_t task);

  // Implementation of the interface.
  MPCallbackEvent Event() override;
  bool CanQueryVariableValues() override;
  double VariableValue(const MPVariable* variable) override;
  void AddCut(const LinearRange& cutting_plane) override;
  void AddLazyConstraint(const LinearRange& lazy_constraint) override;
  double SuggestSolution(
      const absl::flat_hash_map<const MPVariable*, double>& solution) override;
  int64_t NumExploredNodes() override;

  void Update(int32_t where,
              int32_t numdinf,
              const double* dinf_,
              int32_t numiinf,
              const int* iinf_,
              int32_t numliinf,
              const int64_t* liinf_);
  void Update(int32_t numvar,
              double * xx);
  void Update(const char* msg_) {
    msg = msg_;
    ev = MPCallbackEvent::kMessage;
  }
  void Reset();

 private:
  //MSK::Task_t task;
  // current event
  MPCallbackEvent ev;
  // current message if the current event is kMessage
  const char* msg;
  std::vector<double> mosek_variable_values_;
  // NOTE: information items are assigned in callbacks and are valid for the
  // duration of that callback.
  int numdinf, numiinf, numliinf;
  const double * dinf;
  const int32_t* iinf;
  const int64_t* liinf;
};

void MosekMPCallbackContext::Reset() {
  dinf  = nullptr;
  iinf  = nullptr;
  liinf = nullptr;
  msg   = nullptr;
}

void MosekMPCallbackContext::Update(int32_t numvar, double * xx) {
    mosek_variable_values_.resize(numvar);
    std::copy(xx,xx+numvar,mosek_variable_values_.begin());
    ev = MPCallbackEvent::kMipSolution;
}
void MosekMPCallbackContext::Update(int32_t where, int32_t numdinf, const double* dinf_,
                                    int32_t numiinf, const int* iinf_, int32_t numliinf, const int64_t* liinf_) {
  dinf = dinf_;
  iinf = iinf_;
  liinf = liinf_;
  this->numiinf = numiinf;
  this->numliinf = numliinf;
  this->numdinf = numdinf;
  //  kUnknown,
  //  // For regaining control of the main thread in single threaded
  //  applications,
  //  // not for interacting with the solver.
  //  kPolling,
  //  // The solver is currently running presolve.
  //  kPresolve,
  //  // The solver is currently running the simplex method.
  //  kSimplex,
  //  // The solver is in the MIP loop (called periodically before starting a
  //  new
  //  // node).  Useful to early termination.
  //  kMip,
  //  // Called every time a new MIP incumbent is found.
  //  kMipSolution,
  //  // Called once per pass of the cut loop inside each MIP node.
  //  kMipNode,
  //  // Called in each iterate of IPM/barrier method.
  //  kBarrier,
  //  // The solver is about to log out a message, use this callback to capture
  //  it. kMessage,
  //  // The solver is in multi-objective optimization.
  //  kMultiObj,

  std::string_view wherestr(MSK::get_callback_code_name(where));

  if (0 == wherestr.compare("MSK_CALLBACK_BEGIN_DUAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_BEGIN_DUAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_END_DUAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_END_DUAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_IM_PRIMAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_IM_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_IM_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_PRIMAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_UPDATE_DUAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_UPDATE_DUAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX") ||
      0 == wherestr.compare("MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX_BI") ||
      0 == wherestr.compare("MSK_CALLBACK_UPDATE_SIMPLEX"))
  {
      ev = MPCallbackEvent::kSimplex;
  }
  else if (0 == wherestr.compare("MSK_CALLBACK_INTPNT") ||
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_CONIC")
  {
    ev = MPCallbackEvent::kBarrier;
  }
  else if (0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRESOLVE"))
    ev = MPCallbackEvent::kPresolve;
  else if (0 == wherestr.compare("MSK_CALLBACK_BEGIN_MIO"))
    ev = MPCallbackEvent::kMip;
  // The callback function is called from within the basis identification
  // procedure when the primal phase is started.
  else if (0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_BI") ||
           // Begin primal feasibility repair.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_REPAIR") ||
           // Primal sensitivity analysis is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_SENSITIVITY") ||
           // The callback function is called when the primal BI setup is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_PRIMAL_SETUP_BI") ||
           // The basis identification procedure has been started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_BI") ||
           // The callback function is called from within the basis identification
           // procedure when the dual phase is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_DUAL_BI") ||
           // Dual sensitivity analysis is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_DUAL_SENSITIVITY") ||
           // The callback function is called when the dual BI phase is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_DUAL_SETUP_BI") ||
           // The callback function is called when the infeasibility analyzer is
           // started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_INFEAS_ANA") ||
           // The callback function is called when the interior-point optimizer is
           // started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_INTPNT") ||
           // Begin waiting for license.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_LICENSE_WAIT") ||
           // The callback function is called when the optimizer is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_OPTIMIZER") ||
           // Begin QCQO reformulation.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_QCQO_REFORMULATE") ||
           // The callback function is called when root cut generation is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_ROOT_CUTGEN") ||
           // The callback function is called when the simplex optimizer is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_SIMPLEX") ||
           // The callback function is called from within the basis identification
           // procedure when the simplex clean-up phase is started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_SIMPLEX_BI") ||
           // The callback function is called when solution of root relaxation is
           // started.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_SOLVE_ROOT_RELAX") ||
           // Begin conic reformulation.
           0 == wherestr.compare("MSK_CALLBACK_BEGIN_TO_CONIC") ||
           // The callback function is called from within the conic optimizer after the
           // information database has been updated.
           0 == wherestr.compare("MSK_CALLBACK_CONIC") ||
           // The callback function is called from within the dual simplex optimizer.
           0 == wherestr.compare("MSK_CALLBACK_DUAL_SIMPLEX") ||
           // The callback function is called when the basis identification procedure
           // is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_BI") ||
           // The callback function is called when the conic optimizer is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_CONIC") ||
           // The callback function is called from within the basis identification
           // procedure when the dual phase is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_DUAL_BI") ||
           // Dual sensitivity analysis is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_DUAL_SENSITIVITY") ||
           // The callback function is called when the dual BI phase is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_DUAL_SETUP_BI") ||
           // The callback function is called when the infeasibility analyzer is
           // terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_INFEAS_ANA") ||
           // The callback function is called when the interior-point optimizer is
           // terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_INTPNT") ||
           // End waiting for license.
           0 == wherestr.compare("MSK_CALLBACK_END_LICENSE_WAIT") ||
           // The callback function is called when the mixed-integer optimizer is
           // terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_MIO") ||
           // The callback function is called when the optimizer is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_OPTIMIZER") ||
           // The callback function is called when the presolve is completed.
           0 == wherestr.compare("MSK_CALLBACK_END_PRESOLVE") ||
           // The callback function is called from within the basis identification
           // procedure when the primal phase is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_BI") ||
           // End primal feasibility repair.
           0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_REPAIR") ||
           // Primal sensitivity analysis is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_SENSITIVITY") ||
           // The callback function is called when the primal BI setup is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_PRIMAL_SETUP_BI") ||
           // End QCQO reformulation.
           0 == wherestr.compare("MSK_CALLBACK_END_QCQO_REFORMULATE") ||
           // The callback function is called when root cut generation is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_ROOT_CUTGEN") ||
           // The callback function is called when the simplex optimizer is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_SIMPLEX") ||
           // The callback function is called from within the basis identification
           // procedure when the simplex clean-up phase is terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_SIMPLEX_BI") ||
           // The callback function is called when solution of root relaxation is
           // terminated.
           0 == wherestr.compare("MSK_CALLBACK_END_SOLVE_ROOT_RELAX") ||
           // End conic reformulation.
           0 == wherestr.compare("MSK_CALLBACK_END_TO_CONIC") ||
           // The callback function is called from within the basis identification
           // procedure at an intermediate point.
           0 == wherestr.compare("MSK_CALLBACK_IM_BI") ||
           // The callback function is called at an intermediate stage within the conic
           // optimizer where the information database has not been updated.
           0 == wherestr.compare("MSK_CALLBACK_IM_CONIC") ||
           // The callback function is called from within the basis identification
           // procedure at an intermediate point in the dual phase.
           0 == wherestr.compare("MSK_CALLBACK_IM_DUAL_BI") ||
           // The callback function is called at an intermediate stage of the dual
           // sensitivity analysis.
           0 == wherestr.compare("MSK_CALLBACK_IM_DUAL_SENSIVITY") ||
           // The callback function is called at an intermediate point in the dual
           // simplex optimizer.
           0 == wherestr.compare("MSK_CALLBACK_IM_DUAL_SIMPLEX") ||
           // The callback function is called at an intermediate stage within the
           // interior-point optimizer where the information database has not been
           // updated.
           0 == wherestr.compare("MSK_CALLBACK_IM_INTPNT") ||
           // MOSEK is waiting for a license.
           0 == wherestr.compare("MSK_CALLBACK_IM_LICENSE_WAIT") ||
           // The callback function is called from within the LU factorization
           // procedure at an intermediate point.
           0 == wherestr.compare("MSK_CALLBACK_IM_LU") ||
           // The callback function is called at an intermediate point in the
           // mixed-integer optimizer.
           0 == wherestr.compare("MSK_CALLBACK_IM_MIO") ||
           // The callback function is called at an intermediate point in the
           // mixed-integer optimizer while running the dual simplex optimizer.
           0 == wherestr.compare("MSK_CALLBACK_IM_MIO_DUAL_SIMPLEX") ||
           // The callback function is called at an intermediate point in the
           // mixed-integer optimizer while running the interior-point optimizer.
           0 == wherestr.compare("MSK_CALLBACK_IM_MIO_INTPNT") ||
           // The callback function is called at an intermediate point in the
           // mixed-integer optimizer while running the primal simplex optimizer.
           0 == wherestr.compare("MSK_CALLBACK_IM_MIO_PRIMAL_SIMPLEX") ||
           // The callback function is called from within the matrix ordering procedure
           // at an intermediate point.
           0 == wherestr.compare("MSK_CALLBACK_IM_ORDER") ||
           // The callback function is called from within the presolve procedure at an
           // intermediate stage.
           0 == wherestr.compare("MSK_CALLBACK_IM_PRESOLVE") ||
           // The callback function is called from within the basis identification
           // procedure at an intermediate point in the primal phase.
           0 == wherestr.compare("MSK_CALLBACK_IM_PRIMAL_BI") ||
           // The callback function is called at an intermediate stage of the primal
           // sensitivity analysis.
           0 == wherestr.compare("MSK_CALLBACK_IM_PRIMAL_SENSIVITY") ||
           // The callback function is called at an intermediate stage of the conic
           // quadratic reformulation.
           0 == wherestr.compare("MSK_CALLBACK_IM_QO_REFORMULATE") ||
           // The callback is called from within root cut generation at an intermediate
           // stage.
           0 == wherestr.compare("MSK_CALLBACK_IM_ROOT_CUTGEN") ||
           // The callback function is called when the mixed-integer optimizer is
           // restarted.
           0 == wherestr.compare("MSK_CALLBACK_RESTART_MIO") ||
           // The callback function is called while the task is being solved on a
           // remote server.
           0 == wherestr.compare("MSK_CALLBACK_SOLVING_REMOTE") ||
           // The callback function is called from within the basis identification
           // procedure at an intermediate point in the dual phase.
           0 == wherestr.compare("MSK_CALLBACK_UPDATE_DUAL_BI") ||
           // The callback function is called from within the presolve procedure.
           0 == wherestr.compare("MSK_CALLBACK_UPDATE_PRESOLVE") ||
           // The callback function is called from within the basis identification
           // procedure at an intermediate point in the primal phase.
           0 == wherestr.compare("MSK_CALLBACK_UPDATE_PRIMAL_BI"))
    ev = MPCallbackEvent::kPolling;
  else
    ev = MPCallbackEvent::kUnknown;
}

MosekMPCallbackContext::MosekMPCallbackContext(MSK::Task_t task) :
    //task(task),
    ev{} {
  //int numvar = MSK::get_num_var(task);
  //mosek_variable_values_.resize(numvar);
}

int64_t MosekMPCallbackContext::NumExploredNodes() {
  int infidx = MSK::get_iinf_index("MSK_IINF_MIO_NUM_SOLVED_NODES");
  if (0 <= infidx && infidx < numiinf)
    return iinf[infidx];
  else
    return 0;
}

MPCallbackEvent MosekMPCallbackContext::Event() { return ev; }

bool MosekMPCallbackContext::CanQueryVariableValues() {
  const MPCallbackEvent where = Event();
  if (where == MPCallbackEvent::kMipSolution) {
    return true;
  }
  return false;
}

double MosekMPCallbackContext::VariableValue(const MPVariable* variable) {
  CHECK(variable != nullptr);
  CHECK(ev == MPCallbackEvent::kMipSolution || ev == MPCallbackEvent::kMipNode)
      << "You can only call VariableValue at "
      << ToString(MPCallbackEvent::kMipSolution) << " or "
      << ToString(MPCallbackEvent::kMipNode)
      << " but called from: " << ToString(ev);
  return mosek_variable_values_[variable->index()];
}

void MosekMPCallbackContext::AddCut(const LinearRange& cutting_plane) {}
void MosekMPCallbackContext::AddLazyConstraint(
    const LinearRange& lazy_constraint) {}

double MosekMPCallbackContext::SuggestSolution(
    const absl::flat_hash_map<const MPVariable*, double>& solution) {
  return 0;
}

struct MPCallbackWithMosekContext {
  MosekMPCallbackContext* context;
  MPCallback* callback;
  bool* break_solver;
};


void StreamCallbackImpl(MSK::WriteHandle h, const char* msg) {
  MPCallbackWithMosekContext* const callback_with_context =
      static_cast<MPCallbackWithMosekContext*>(h);

  CHECK(callback_with_context != nullptr);
  CHECK(callback_with_context->context != nullptr);
  callback_with_context->context->Update(msg);

  if (callback_with_context->callback) {
    callback_with_context->callback->RunCallback(
        callback_with_context->context);
  }

  callback_with_context->context->Reset();
}

// NOTE(user): This function must have this exact API, because we are passing
// it to Mosek as a callback.
int CallbackImpl(MSK::CallbackHandle h,
                        int32_t where,
                        int32_t numdinf,
                        const double* dinf,
                        int32_t numiinf,
                        const int* iinf,
                        int32_t numliinf,
                        const int64_t* liinf)
{
  MPCallbackWithMosekContext* const callback_with_context =
      static_cast<MPCallbackWithMosekContext*>(h);
  CHECK(callback_with_context != nullptr);
  CHECK(callback_with_context->context != nullptr);

  if (callback_with_context->callback) {
    callback_with_context->context->Update(where, numdinf, dinf, numiinf, iinf, numliinf, liinf);
    callback_with_context->callback->RunCallback(
        callback_with_context->context);
  }
  callback_with_context->context->Reset();
  return callback_with_context->break_solver[0] ? 1 : 0;
}

}  // namespace

#if 0  // TODO
MPSolutionResponse MosekInterface::DirectlySolveProto(LazyMutableCopy<MPModelRequest> request,
                                      std::atomic<bool>* interrupt) override {
  DCHECK_EQ(interrupt, nullptr);
  const bool log_error = request->enable_internal_solver_output();

  // Here we reuse the Mosek environment to support single-use license that
  // forbids creating a second environment if one already exists.
  return ConvertStatusOrMPSolutionResponse(
      log_error, MosekSolveProto(std::move(request), global_env_));
}
#endif

void MosekInterface::CheckedMosekCall(MSK::ResCode r) const {
  ::operations_research::CheckedMosekCall(task_, r);
}

// Creates a LP/MIP instance with the specified name and minimization objective.
MosekInterface::MosekInterface(MPSolver* const solver, bool mip)
  : MPSolverInterface(solver),
    task_(nullptr),
    mip_(false),
    ptask_(nullptr, DeleteTask),
    break_solver_(false)
{
  if (nullptr == ((task_ = MSK::new_task())))
    throw std::bad_alloc();

  ptask_.reset(task_);
  CheckedMosekCall(MSK::put_task_name(task_, solver_->Name().c_str()));
  MSK::put_objective_sense(task_, maximize_ ? MSK::ObjSense::MAXIMIZE : MSK::ObjSense::MINIMIZE);

  CheckedMosekCall(MSK::get_domain_rzero(task_,  &domidx_rzero));
  CheckedMosekCall(MSK::get_domain_r(task_,      &domidx_rfree));
  CheckedMosekCall(MSK::get_domain_rplus(task_,  &domidx_rplus));
  CheckedMosekCall(MSK::get_domain_rminus(task_, &domidx_rminus));
}

MosekInterface::~MosekInterface() {}

// ------ Model modifications and extraction -----

void MosekInterface::Reset() {
  decltype(ptask_) new_task(MSK::new_task(),DeleteTask);
  if (! new_task) {
    throw std::bad_alloc();
  }
  else {
    mp_cons_to_mosek_cons_.clear();

    MosekCloneParameters(new_task.get(), task_);

    task_ = new_task.get();
    ptask_ = std::move(new_task);
    if (MSK::RES_OK != MSK::append_rows(task_,1) ||
        MSK::RES_OK != MSK::put_objective_row(task_,0))
      throw std::bad_alloc();
  }

  ResetExtractionInformation();
}

void MosekInterface::SetOptimizationDirection(bool maximize) {
  InvalidateSolutionSynchronization();
  MSK::put_objective_sense(task_, maximize ? MSK::ObjSense::MAXIMIZE : MSK::ObjSense::MINIMIZE);
}

void MosekInterface::SetVariableBounds(int var_index, double lb, double ub) {
  InvalidateSolutionSynchronization();
  CheckedMosekCall(MSK::put_var_bound(task_,var_index,lb,ub));
}

void MosekInterface::SetVariableInteger(int index, bool integer) {
  InvalidateSolutionSynchronization();

  CheckedMosekCall(MSK::put_var_type(task_, index, integer ? MSK::VariableType::INTEGER : MSK::VariableType::CONTINUOUS));
}

void MosekInterface::SetConstraintBounds(int index, double lb, double ub) {
  InvalidateSolutionSynchronization();
  auto [i,afei] = mp_cons_to_mosek_cons_[index];
  if (i >= 0) {
    double rhs[2] = {lb,ub};
    int64_t doms[2] = {std::isfinite(lb) ? domidx_rplus : domidx_rfree ,std::isfinite(ub) ? domidx_rminus : domidx_rfree };
    int64_t afeidxs[] = { afei, afei };

    if (lb <= ub && lb >= ub) {
      doms[0] = domidx_rzero;
      doms[1] = domidx_rfree;
    }
    CheckedMosekCall(MSK::put_con_slice(task_,i,2,2,doms,afeidxs,rhs));
  } else {
    int64_t djci = -i-1;
    int64_t afeidxs[4] = {afei, afei, afei + 1, afei + 1};
    double b[4] = {0.0, 1.0, lb, ub};
    int64_t termsize[2]{1, 3};
    int64_t domidxs[4]{domidx_rzero, domidx_rzero, domidx_rplus, domidx_rminus};

    if (lb <= ub && lb >= ub) {
      domidxs[2] = domidx_rzero;
      domidxs[3] = domidx_rfree;
    } else {
      if (lb < 0 && !std::isfinite(lb)) domidxs[2] = domidx_rfree;
      if (ub > 0 && !std::isfinite(ub)) domidxs[3] = domidx_rfree;
    }

    CheckedMosekCall(MSK::put_djc(task_, djci, /*num row*/ 4, /*num domain*/ 4, /*num term*/ 2, domidxs, termsize, afeidxs, b));
  }
}

// Ordinary linear constraint are added as two constraints, one for upper bound,
// one for lower bound.
//
// Indicator constraints are added as a disjunctive constraints, so X => lb < AX < ub becomes:
//
//   X = 0
// OR
//   X = 1
//   lb <= Ax
//   ub >= Ax
void MosekInterface::AddRowConstraint(MPConstraint* const ct) {
  int64_t conidx = MSK::get_num_con(task_);
  int64_t afei   = MSK::get_num_row(task_);

  CheckedMosekCall(MSK::append_rows(task_, 1));
  mp_cons_to_mosek_cons_.push_back(std::make_tuple(conidx,afei));

  double lb = ct->lb();
  double ub = ct->ub();
  const std::string& name = ct->name();

  int64_t afeidxs[2] = { afei,afei };
  int64_t domidxs[2] = { std::isfinite(lb) ? domidx_rplus : domidx_rfree, std::isfinite(ub) ? domidx_rminus : domidx_rfree };
  double  rhs[2]     = { lb, ub };

  CheckedMosekCall(MSK::append_cons(task_,2,));
  CheckedMosekCall(MSK::put_con_slice(task_,conidx,2,2,domidxs,afeidxs,rhs));

  std::vector<double> cof;
  cof.reserve(ct->coefficients_.size());
  std::vector<int> subj;
  subj.reserve(cof.size());
  for (const auto& it : ct->terms()) {
    subj.push_back(it.first->index());
    cof.push_back(it.second);
  }
  if (name.size() > 0) {
    CheckedMosekCall(MSK::put_con_name(task_, conidx, name.c_str()));
    CheckedMosekCall(MSK::put_con_name(task_, conidx+1, name.c_str()));
  }

  CheckedMosekCall(MSK::put_row(task_, afei, subj.size(), subj.data(), cof.data()));
}

bool MosekInterface::AddIndicatorConstraint(MPConstraint* const ct) {
  int64_t
    djci = MSK::get_num_djc(task_),
    afei = MSK::get_num_row(task_);
  CheckedMosekCall(MSK_appenddjcs(task_, 1));
  CheckedMosekCall(MSK_appendafes(task_, 2));
  mp_cons_to_mosek_cons_.push_back(std::make_tuple(-1-djci,afei));

  auto indvar = ct->indicator_variable();

  CHECK(indvar != nullptr);
  //CheckedMosekCall(MSK_putvartype(task_, indvar->index(), MSK_VAR_TYPE_INT));

  //CheckedMosekCall(
  //    MSK_putvarbound(task_, indvar->index(), MSK_BK_RA, 0.0, 1.0));

  const std::string& name = ct->name();
  if (name.size() > 0)
    CheckedMosekCall(MSK::put_djc_name(task_, djci, name.c_str()));

  {
    double lb = ct->lb();
    double ub = ct->ub();

    int64_t domidxs[4] = {domidx_rzero, domidx_rzero,
                          std::isfinite(lb) ? domidx_rplus : domidx_rfree,
                          std::isfinite(ub) ? domidx_rminus : domidx_rfree};
    int64_t afeidxs[4] = {afei, afei, afei + 1, afei + 1};
    double b[4] = {0.0, 1.0, lb, ub};
    int64_t termsize[2] = {1, 3};

    if (lb <= ub && lb >= ub) {
      domidxs[2] = domidx_rzero;
      domidxs[3] = domidx_rfree;
    }

    CheckedMosekCall(
        MSK::put_djc(task_, djci, 4,4,2, domidxs,termsize,afeidxs,b));
  }
  {
    std::vector<double> cof;
    cof.reserve(ct->coefficients_.size());
    std::vector<int> subj;
    subj.reserve(cof.size());
    for (const auto& it : ct->terms()) {
      subj.push_back(it.first->index());
      cof.push_back(it.second);
    }
    CheckedMosekCall(
        MSK::put_row(task_, afei + 1, subj.size(), subj.data(), cof.data()));
  }
  {
    double c = 1.0;
    int indvarj = indvar->index();
    CheckedMosekCall(MSK::put_row(task_, afei, 1, &indvarj, &c));
  }

  return true;
}

void MosekInterface::AddVariable(MPVariable* const var) {
  int j = MSK::get_num_var(task_);
  CheckedMosekCall(MSK_appendvars(task_, 1));
  double lb = var->lb();
  double ub = var->ub();

  const std::string& name = var->name();

  CheckedMosekCall(MSK::put_var_bound(task_, j, lb, ub));
  if (var->integer())
    CheckedMosekCall(MSK::put_var_type(task_, j, MSK::VariableType::INTEGER));
  if (name.size() > 0) CheckedMosekCall(MSK::put_var_name(task_, j, name.c_str()));
}

void MosekInterface::SetCoefficient(MPConstraint* const constraint,
                                    const MPVariable* const variable,
                                    double new_value, double old_value) {
  InvalidateSolutionSynchronization();
  auto [i,afei] = mp_cons_to_mosek_cons_[constraint->index()];
  if (i >= 0) {
    CheckedMosekCall(MSK::put_ijc(task_, afei, variable->index(), new_value));
  } else {
    int64_t djci = -i-1;

    CheckedMosekCall(
        MSK_putafefentry(task_, afei+1, variable->index(), new_value));
  }
}

// Question: Is an indicator constraint ever cleared? What exactly does that
// mean?
void MosekInterface::ClearConstraint(MPConstraint* const constraint) {
  InvalidateSolutionSynchronization();
  // TODO: Cleanup if afe nonzeros etc?
  auto [i,afei] = mp_cons_to_mosek_cons_[constraint->index()];
  if (i >= 0) {
    int64_t afeidxs[2] = {afei,afei};
    int64_t domidxs[2] = {domidx_rfree,domidx_rfree};
    double rhs[2] = {0.0,0.0};
    CheckedMosekCall(MSK::put_con_slice(task_,i,2,2,domidxs,afeidxs,rhs);
  } else {
    int64_t djci = -coni - 1;
    CheckedMosekCall(
        MSK::put_djc(task_, djci, 0, 0, 0, nullptr, nullptr, nullptr, nullptr));
  }
}

void MosekInterface::SetObjectiveCoefficient(const MPVariable* const variable,
                                             double coefficient) {
  InvalidateSolutionSynchronization();
  CheckedMosekCall(MSK::put_ijc(task_, 0, variable->index(), coefficient));
}

void MosekInterface::SetObjectiveOffset(double value) {
  InvalidateSolutionSynchronization();

  CheckedMosekCall(MSK_putcfix(task_, value));
}

void MosekInterface::ClearObjective() {
  InvalidateSolutionSynchronization();
  int numvar;
  CheckedMosekCall(MSK_getnumvar(task_, &numvar));
  for (int i = 0; i < numvar; ++i) MSK_putcj(task_, i, 0.0);
  MSK::put_row_g(task_, 0, 0.0);
}

// ------ Query statistics on the solution and the solve ------

int64_t MosekInterface::iterations() const {
  if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfIterations;
  // TODO: Iter Count
  int32_t psim_iter = 0, dsim_iter = 0, intpnt_iter = 0;
  int64_t mio_intpnt_iter = 0, mio_simplex_iter = 0;
  int32_t index;
  if (0 <= (index = MSK::get_iinf_index("MSK_IINF_SIM_DUAL_ITER")))
    MSK::get_iinf(task_,index,&psim_iter);
  if (0 <= (index = MSK::get_iinf_index("MSK_IINF_SIM_PRIMAL_ITER")))
    MSK::get_iinf(task_,index,&dsim_iter);
  if (0 <= (index = MSK::get_iinf_index("MSK_IINF_INTPNT_ITER")))
    MSK::get_iinf(task_,index,&intpnt_iter);
  if (0 <= (index = MSK::get_liinf_index("MSK_LIINF_MIO_INTPNT_ITER")))
    MSK::get_liinf(task_,index,&mio_intpnt_iter);
  if (0 <= (index = MSK::get_liinf_index("MSK_LIINF_MIO_SIMPLEX_ITER")))
    MSK::get_liinf(task_,index,&mio_simplex_iter);

  if (intpnt_iter > 0)
    return intpnt_iter;
  else if (psim_iter+dsim_iter > 0)
    return psim_iter+dsim_iter;
  else if (mio_simplex_iter > 0)
    return mio_simplex_iter;
  else if (mio_intpnt_iter > 0)
    return mio_intpnt_iter;
  else
    return 0;
}

int64_t MosekInterface::nodes() const {
  if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfNodes;
  int nnodes = 0;
  int index = MSK::get_iinf_index("MSK_IINF_MIO_NUM_SOLVED_NODES");
  if (index >= 0)
      MSK::get_iinf(task_, nidex, &nnodes));
  return nnodes;
}

// Returns the basis status of a row.
MPSolver::BasisStatus MosekInterface::row_status(int constraint_index) const {
  auto [coni,afei] = mp_cons_to_mosek_cons_[constraint_index];
  if (coni < 0) {
    LOG(DFATAL) << "Basis status only available for continuous problems.";
  }

  int nsol = MSK::get_num_solutions(task_);
  // locate a basic solutoin
  MSK::SolType st;
  for (int soli = 0; soli < nsol; ++soli) {
    MSK::get_solution_type(task_,soli,&st);
    if (st == MSK::SolType::BASIC) {
      int lowbas,uprbas;
      int lowbind,uprbind;
      MSK::get_solution_basic_con(task_,soli,coni,&lowbas);
      MSK::get_solution_basic_con(task_,soli,coni+1,&uprbas);
      MSK::get_solution_sta_con(task_,soli,coni,&lowbind);
      MSK::get_solution_sta_con(task_,soli,coni+1,&uprbind);

      if (lowbas || uprbas) return MPSolver::BASIC;
      else if (lowbind && uprbind) return MPSolver::FIXED_VALUE;
      else if (lowbind) return MPSolver::AT_LOWER_BOUND;
      else if (uprbind) return MPSolver::AT_UPPER_BOUND;
      else              return MPSolver::FREE;
    }
  }

  LOG(DFATAL)
      << "Basis status only available when a basis solution has been found.";
  return MPSolver::FREE;
}

// Returns the basis status of a column.
MPSolver::BasisStatus MosekInterface::column_status(int j) const {
  int nsol = MSK::get_num_solutions(task_);
  // locate a basic solutoin
  MSK::SolType st;
  for (int soli = 0; soli < nsol; ++soli) {
    MSK::get_solution_type(task_,soli,&st);
    if (st == MSK::SolType::BASIC) {
      int bas;
      int lowbind,uprbind;
      MSK::get_solution_basic_xj(task_,soli,j,&bas);
      MSK::get_solution_sta_xj(task_,soli,j,&lowbind,&uprbind);

      if       (bas)    return MPSolver::BASIC;
      else if (lowbind && uprbind) return MPSolver::FIXED_VALUE;
      else if (lowbind) return MPSolver::AT_LOWER_BOUND;
      else if (uprbind) return MPSolver::AT_UPPER_BOUND;
      else              return MPSolver::FREE;
    }
  }

  LOG(DFATAL)
      << "Basis status only available when a basis solution has been found.";
  return MPSolver::FREE;
}

// Extracts new variables.
void MosekInterface::ExtractNewVariables() {
  int numvar = MSK::get_num_var(task_);
  auto total_num_vars = solver_->variables_.size();
  if (total_num_vars > numvar) {
    int numnewvars = numvar - total_num_vars;
    CheckedMosekCall(MSK::append_vars(task_, numnewvars));
    auto& obj = solver_->Objective();

    for (int j = numvar; numvar < total_num_vars; ++j) {
      auto var = solver_->variables_[j];
      set_variable_as_extracted(j, true);
      CheckedMosekCall(MSK::put_var_bound(task_, j, var->lb(), var->ub()));
      if (var->integer()) {
        CheckedMosekCall(MSK::put_var_type(task_, j, MSK::VariableType::INTEGER));
      }

      double cj = obj.GetCoefficient(var);
      if (cj > 0.0 || cj < 0.0) {
        CheckedMosekCall(MSK::put_ijc(task_, 0,j, cj));
      }

      for (int i = 0; i < mp_cons_to_mosek_cons_.size(); ++i) {
        auto [coni,afei] = mp_cons_to_mosek_cons_[i];
        const MPConstraint* ct = solver_->constraints()[i];
        if (coni >= 0) {
          for (const auto& item : ct->coefficients_) {
            if (item.first->index() >= numvar) {
              CheckedMosekCall(
                  MSK::put_ijc(task_, afei, item.first->index(), item.second));
            }
          }
        } else {
          auto djci = -coni- 1;
          for (const auto& item : ct->coefficients_) {
            if (item.first->index() >= numvar) {
              CheckedMosekCall(MSK::put_ijc(
                  task_, afei + 1, item.first->index(), item.second));
            }
          }
        }
      }
    }
  }
}

void MosekInterface::ExtractNewConstraints() {
  int total_num_rows = solver_->constraints_.size();
  if (mp_cons_to_mosek_cons_.size() < total_num_rows) {
    // Add each new constraint.
    for (int row = last_constraint_index_; row < total_num_rows; ++row) {
      MPConstraint* const ct = solver_->constraints_[row];
      set_constraint_as_extracted(row, true);
      AddRowConstraint(ct);
    }
  }
}

void MosekInterface::ExtractObjective() {
  MSK::put_objective_sense(task_, maximize_ ? MSK::ObjSense::MAXIMIZE : MSK::ObjSense::MINIMIZE));
  const auto& obj = solver_->Objective();
  CheckedMosekCall(MSK::put_row_g(task_,0,obj.offset()));
}

// ------ Parameters  -----

void MosekInterface::SetParameters(const MPSolverParameters& param) {
  SetCommonParameters(param);
  if (mip_) {
    SetMIPParameters(param);
  }
}

bool MosekInterface::SetSolverSpecificParametersAsString(
    const std::string& parameters) {
  std::string_view data(parameters);
  std::string key, value;
  while (data.size() > 0) {
    auto p = data.find('\n');
    if (p == std::string_view::npos) p = data.size();
    std::string_view line = data.substr(0, p);

    auto eq_pos = line.find('=');
    if (eq_pos != std::string_view::npos) {
      key.clear();
      value.clear();
      key += line.substr(0, eq_pos);
      value += line.substr(eq_pos + 1);

      if (MSK_RES_OK != MSK::put_parameter_str(task_, key.c_str(), value.c_str())) {
        LOG(WARNING) << "Failed to set parameters '" << key << "' to '" << value
                     << "'";
      }
    }

    data = data.substr(p + 1);
  }
  return true;
}

void MosekInterface::SetRelativeMipGap(double value) {
  CheckedMosekCall(MSK::put_double_param(task_, "MSK_DPAR_MIO_REL_GAP_CONST", value));
}

void MosekInterface::SetPrimalTolerance(double value) {
  CheckedMosekCall(MSK::put_double_parameter(task_, "MSK_DPAR_INTPNT_TOL_PFEAS", value));
  CheckedMosekCall(MSK::put_double_parameter(task_, "MSK_DPAR_BASIS_TOL_X", value));
}

void MosekInterface::SetDualTolerance(double value) {
  CheckedMosekCall(MSK::put_double_parameter(task_, "MSK_DPAR_INTPNT_TOL_DFEAS", value));
  CheckedMosekCall(MSK::put_double_parameter(task_, "MSK_DPAR_BASIS_TOL_S", value));
}

void MosekInterface::SetPresolveMode(int value) {
  switch (value) {
    case MPSolverParameters::PRESOLVE_OFF: {
      CheckedMosekCall(MSK::put_parameter_str(task_, "MSK_IPAR_PRESOLVE_USE", "MSK_OFF"));
      break;
    }
    case MPSolverParameters::PRESOLVE_ON: {
      CheckedMosekCall(MSK::put_parameter_str(task_, "MSK_IPAR_PRESOLVE_USE", "MSK_ON"));
      break;
    }
    default: {
      SetIntegerParamToUnsupportedValue(MPSolverParameters::PRESOLVE, value);
    }
  }
}

// Sets the scaling mode.
void MosekInterface::SetScalingMode(int value) {
  switch (value) {
    case MPSolverParameters::SCALING_OFF:
      CheckedMosekCall(
          MSK_putintparam(task_, MSK_IPAR_INTPNT_SCALING, MSK_SCALING_NONE));
      CheckedMosekCall(
          MSK_putintparam(task_, MSK_IPAR_SIM_SCALING, MSK_SCALING_NONE));
      break;
    case MPSolverParameters::SCALING_ON:
      CheckedMosekCall(
          MSK_putintparam(task_, MSK_IPAR_INTPNT_SCALING, MSK_SCALING_FREE));
      CheckedMosekCall(
          MSK_putintparam(task_, MSK_IPAR_SIM_SCALING, MSK_SCALING_FREE));
      break;
    default:
      // Leave the parameters untouched.
      break;
  }
}

void MosekInterface::SetLpAlgorithm(int value) {
  switch (value) {
    case MPSolverParameters::DUAL:
      CheckedMosekCall(MSK_putintparam(task_, MSK_IPAR_OPTIMIZER,
                                       MSK_OPTIMIZER_DUAL_SIMPLEX));
      break;
    case MPSolverParameters::PRIMAL:
      CheckedMosekCall(MSK_putintparam(task_, MSK_IPAR_OPTIMIZER,
                                       MSK_OPTIMIZER_PRIMAL_SIMPLEX));
      break;
    case MPSolverParameters::BARRIER:
      CheckedMosekCall(
          MSK_putintparam(task_, MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT));
      break;
    default:
      SetIntegerParamToUnsupportedValue(MPSolverParameters::LP_ALGORITHM,
                                        value);
  }
}

int MosekInterface::SolutionCount() {
  int soldef;
  MSK_solutiondef(task_, MSK_SOL_ITG, &soldef);
  if (soldef) return 1;
  MSK_solutiondef(task_, MSK_SOL_BAS, &soldef);
  if (soldef) return 1;
  MSK_solutiondef(task_, MSK_SOL_ITR, &soldef);
  if (soldef) return 1;
  return 0;
}

MPSolver::ResultStatus MosekInterface::Solve(const MPSolverParameters& param) {
  WallTimer timer;
  timer.Start();


  // Set log level.
  CheckedMosekCall(MSK_putintparam(task_, MSK_IPAR_LOG, quiet_ ? 0 : 10));

  ExtractModel();
  // Sync solver.
  VLOG(1) << absl::StrFormat("Model built in %s.",
                             absl::FormatDuration(timer.GetDuration()));

  int numvar;
  MSK_getnumvar(task_, &numvar);
  bool ismip = false;
  for (int j = 0; j < numvar && ! ismip; ++j) {
    MSKvariabletypee vt;
    MSK_getvartype(task_,j,&vt); ismip = ismip && vt == MSK_VAR_TYPE_INT;
  }
  // Set solution hints. Currently this only affects integer solution.
  if (solver_->solution_hint_.size() > 0) {
    std::vector<double> xx(numvar);
    for (const std::pair<const MPVariable*, double>& p :
         solver_->solution_hint_) {
      xx[p.first->index()] = p.second;
    }
    MSK_putxx(task_, MSK_SOL_ITG, xx.data());
  }

  // Time limit.
  if (solver_->time_limit() != 0) {
    VLOG(1) << "Setting time limit = " << solver_->time_limit() << " ms.";
    CheckedMosekCall(MSK_putdouparam(task_, MSK_DPAR_OPTIMIZER_MAX_TIME,
                                     solver_->time_limit_in_secs()));
  }

  // We first set our internal MPSolverParameters from 'param' and then set
  // any user-specified internal solver parameters via
  // solver_specific_parameter_string_.
  // Default MPSolverParameters can override custom parameters (for example for
  // presolving) and therefore we apply MPSolverParameters first.
  SetParameters(param);
  solver_->SetSolverSpecificParametersAsString(
      solver_->solver_specific_parameter_string_);

  // remove any pre-existing solution in task that are not relevant for the
  // result.
  MSK_putintparam(task_, MSK_IPAR_REMOVE_UNUSED_SOLUTIONS, MSK_ON);

  // Solve
  timer.Restart();

  MSKrescodee trm;
  {
    MosekMPCallbackContext mosek_context(task_);
    MPCallback* cb = callback_;
    MPCallbackWithMosekContext mp_callback_with_context = {
        .context = &mosek_context,
        .callback = cb,
        .break_solver = &break_solver_};

    auto _remove_cb = absl::MakeCleanup([&]() {
      MSK_linkfunctotaskstream(task_, MSK_STREAM_LOG, nullptr, nullptr);
      MSK_putcallbackfunc(task_, nullptr, nullptr);
    });
    MSK_putcallbackfunc(task_, CallbackImpl, &mp_callback_with_context);
    MSK_linkfunctotaskstream(task_, MSK_STREAM_LOG, &mp_callback_with_context,
                             StreamCallbackImpl);
    CheckedMosekCall(MSK_optimizetrm(task_, &trm));
  }

  VLOG(1) << absl::StrFormat("Solved in %s.",
                             absl::FormatDuration(timer.GetDuration()));
  // Get the status.
  MSKprostae prosta = (MSKprostae)-1;
  MSKsolstae solsta = (MSKsolstae)-1;
  MSKsoltypee whichsol;
  bool soldefined = true;
  {
    int soldef;
    whichsol = MSK_SOL_ITG;
    MSK_solutiondef(task_, whichsol, &soldef);
    if (!soldef) {
      whichsol = MSK_SOL_BAS;
      MSK_solutiondef(task_, whichsol, &soldef);
    }
    if (!soldef) {
      whichsol = MSK_SOL_ITR;
      MSK_solutiondef(task_, whichsol, &soldef);
    }
    soldefined = soldef != 0;
  }

  if (soldefined) {
    MSK_getprosta(task_, whichsol, &prosta);
    MSK_getsolsta(task_, whichsol, &solsta);
  }

  VLOG(1) << absl::StrFormat("Solution status %d.\n", prosta);
  const int solution_count = SolutionCount();

  if (!soldefined) {
    result_status_ = MPSolver::NOT_SOLVED;
  } else if (solsta == MSK_SOL_STA_OPTIMAL ||
             solsta == MSK_SOL_STA_INTEGER_OPTIMAL) {
    result_status_ = MPSolver::OPTIMAL;
  } else if (solsta == MSK_SOL_STA_PRIM_AND_DUAL_FEAS) {
    result_status_ = MPSolver::FEASIBLE;
  } else if (prosta == MSK_PRO_STA_PRIM_INFEAS) {
    result_status_ = MPSolver::INFEASIBLE;
  } else if (prosta == MSK_PRO_STA_DUAL_INFEAS) {
    result_status_ = MPSolver::UNBOUNDED;
  } else if (prosta == MSK_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED) {
    // TODO(user): We could introduce our own "infeasible or
    // unbounded" status.
    result_status_ = MPSolver::INFEASIBLE;
  } else {
    result_status_ = MPSolver::NOT_SOLVED;
  }

  // Get best objective bound value
  if (whichsol == MSK_SOL_ITG &&
      (result_status_ == MPSolver::FEASIBLE ||
       result_status_ == MPSolver::OPTIMAL)) {
    MSK_getdouinf(task_, MSK_DINF_MIO_OBJ_BOUND, &best_objective_bound_);
    VLOG(1) << "best bound = " << best_objective_bound_;
  }

  if (solution_count > 0 && (result_status_ == MPSolver::FEASIBLE ||
                             result_status_ == MPSolver::OPTIMAL)) {
    // Get the results.
    MSK_getprimalobj(task_, whichsol, &objective_value_);
    VLOG(1) << "objective = " << objective_value_;

    std::vector<double> xx(numvar);
    CheckedMosekCall(MSK_getxx(task_, whichsol, xx.data()));
    {
      for (int i = 0; i < solver_->variables_.size(); ++i) {
        MPVariable* const var = solver_->variables_[i];
        var->set_solution_value(xx[i]);
        VLOG(3) << var->name() << ", value = " << xx[i];
      }
    }
    if (whichsol != MSK_SOL_ITG) {
      {
        std::vector<double> slx(numvar);
        std::vector<double> sux(numvar);

        CheckedMosekCall(MSK_getslx(task_, whichsol, slx.data()));
        CheckedMosekCall(MSK_getsux(task_, whichsol, sux.data()));

        for (int i = 0; i < solver_->variables_.size(); ++i) {
          MPVariable* const var = solver_->variables_[i];
          var->set_reduced_cost(slx[i] - sux[i]);
          VLOG(4) << var->name() << ", reduced cost = " << (slx[i] - sux[i]);
        }
      }

      {
        size_t numcon = mp_cons_to_mosek_cons_.size();
        std::vector<double> y(numcon);

        CheckedMosekCall(MSK_gety(task_, whichsol, y.data()));

        for (int i = 0; i < solver_->constraints_.size(); ++i) {
          MPConstraint* const ct = solver_->constraints_[i];
          auto coni = mp_cons_to_mosek_cons_[ct->index()];
          if (coni >= 0) {
            ct->set_dual_value(y[coni]);
            VLOG(4) << "row " << ct->index() << ", dual value = " << y[coni];
          }
        }
      }
    }
  }

  sync_status_ = SOLUTION_SYNCHRONIZED;
  return result_status_;
}

// Select next solution and assign all solution values to variables.
bool MosekInterface::NextSolution() { return false; }

void MosekInterface::Write(const std::string& filename) {
  // if (sync_status_ == MUST_RELOAD) {
  //   Reset();
  // }
  // ExtractModel();
  VLOG(1) << "Writing Mosek Task file \"" << filename << "\".";
  MSKrescodee r = MSK_writedata(task_, filename.c_str());
  if (MSK_RES_OK != r) {
    auto lasterr = MosekLastError(task_);
    LOG(WARNING) << "Failed to write Task. Error (" << lasterr.second
                 << "): " << lasterr.first;
  }
}

MPSolverInterface* BuildMosekInterface(bool mip, MPSolver* const solver) {
  return new MosekInterface(solver, mip);
}

void MosekInterface::SetCallback(MPCallback* mp_callback) {
  callback_ = mp_callback;
}

namespace {
    const void * const kRegisterMosekLp ABSL_ATTRIBUTE_UNUSED = [] {
    MPSolverInterfaceFactoryRepository::GetInstance()->Register(
        [](MPSolver*solver) {return new MosekInterface(solver,false)},
        MPSolver::MOSEK_LINEAR_PROGRAMMING,
        []() { return MSK120::initialize_library(); }),

    MPSolverInterfaceFactoryRepository::GetInstance()->Register(
        [](MPSolver*solver) {return new MosekInterface(solver,true)},
        MPSolver::MOSEK_MIZED_INTEGER_PROGRAMMING,
        []() { return MSK120::initialize_library(); }),

    }

}

}  // namespace operations_research
