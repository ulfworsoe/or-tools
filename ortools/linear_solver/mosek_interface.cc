// Copyright 2010-2024 Google LLg
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
#include <strings.h>
#include <string_view>
#if defined(USE_MOSEK)

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/die_if_null.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "ortools/base/logging.h"
#include "ortools/base/timer.h"
#include "ortools/linear_solver/linear_solver.h"
#include "ortools/linear_solver/linear_solver_callback.h"
#include "ortools/linear_solver/proto_solver/proto_utils.h"
#include "ortools/util/lazy_mutable_copy.h"
#include "ortools/util/time_limit.h"

#include "mosek.h"

namespace operations_research {

  class MosekInterface : public MPSolverInterface {
  public:

    explicit MosekInterface(MPSolver* solver, bool mip);
    ~MosekInterface() override;

    void SetOptimizationDirection(bool maximize) override;

    // ----- Solve -----
  // Solves the problem using the parameter values specified.
  MPSolver::ResultStatus Solve(const MPSolverParameters& param) override;



  // ----- Directly solve -----
  // Mosek should support being interrupted, but for now we'll only support non-interupted solves.
  bool SupportsDirectlySolveProto(std::atomic<bool>* interrupt) const override {
    return interrupt == nullptr;
  }

  // Solve model dirctly, bypassing protobuffers
  MPSolutionResponse DirectlySolveProto(LazyMutableCopy<MPModelRequest> request,
                                        std::atomic<bool>* interrupt) override;

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
  void BranchingPriorityChangedForVariable(int var_index) override;

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
  bool IsContinuous() const override { return IsLP(); }
  bool IsLP() const override { return !mip_; }
  bool IsMIP() const override { return mip_; }

  void ExtractNewVariables() override;
  void ExtractNewConstraints() override;
  void ExtractObjective() override;
  
  std::string SolverVersion() const override {
    int major, minor, rev;
    
    MSK_getversion(&major, &minor, &rev);
    return absl::StrFormat("Mosek library version %d.%d.%d\n", major, minor,
                           rev);
  }

  bool InterruptSolve() override;

  void* underlying_solver() override { return reinterpret_cast<void*>(model_); }

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

 private:
  static MSKboundkeye bk_from_bounds(double lb, double ub);

  void CheckedMosekCall(MSKrescodee r) const;

  //----- Variables -----

  // The underlying MOSEK task, which is kept updated with all Add* calls.
  MSKtask_t task_;
  bool break_solver_;
  std::unique_ptr<MSKtask_t,decltype(MSK_deletetask)*> ptask_;

  bool mip_;
  //int current_solution_index_;
  // mp callback function
  MPCallback* callback_ = nullptr;
  // Has length equal to the number of MPConstraints in
  // MPSolverInterface::solver_. Non-negative values are indexes of the
  // corresponding linear constraint in Mosek, negative index i means
  // disjunctive constraint (-i-1), used for indicator constraints.
  std::vector<ssize_t> mp_cons_to_mosek_cons_;
  std::vector<ssize_t> indcon_afeidx;

  int64_t domidx_rfree;
  int64_t domidx_rzero;
  int64_t domidx_rplus;
  int64_t domidx_rminus;

  mutable absl::Mutex hold_interruptions_mutex_;
}; // MosekInterface

namespace {

std::pair<std::string,MSKrescodee> MosekLastError(MSKtask_t task) {
  int64_t lastmsgsize;
  std::vector<char> lastmsg();
  MSKrescodee lastr;
  MSKrescodee r = MSK_getlasterror64(task,&lastr,0,&lastmsgsize,nullptr);
  if (MSK_RES_OK == r) {
    lastmsg.resize(lastmsgsize+1);
    MSK_getlasterror64(task,&lastr,lastmsgsize,lastmsg.data());
    return std::make_pair(lastmsg.data(), lastr);
  }

  return std:make_pair("",MSK_RES_OK);
}

void CheckedMosekCall(MSKtask_t task, MSKrescodee r) {  
  CHECK_EQ(MSK_RES_OK, r)
      << "Mosek Error " << r << ": " << MosekLastError(task).second;
}






bool MosekInterface::InterruptSolve() {
  const absl::MutexLock lock(&hold_interruptions_mutex_);
  break_solver_ = true;

  return true;
}

// This class provides a means of interacting with the task from the callback
// function.
class MosekMPCallbackContext : public MPCallbackContext {
 public:
  MosekMPCallbackContext(MSKtask_t task)

  // Implementation of the interface.
  MPCallbackEvent Event() override;
  bool CanQueryVariableValues() override;
  double VariableValue(const MPVariable* variable) override;
  void AddCut(const LinearRange& cutting_plane) override;
  void AddLazyConstraint(const LinearRange& lazy_constraint) override;
  double SuggestSolution(const absl::flat_hash_map<const MPVariable*, double>& solution) override;
  int64_t NumExploredNodes() override;

 private:
  // current event
  MPCallbackEvent ev;
  // current message if the current event is kMessage
  const char * msg;
  std::vector<double> mosek_variable_values_;
};

MosekMPCallbackContext::MosekMPCallbackContext(MSKtask_t task) : task(task) {
  int numvar; MSK_getnumvar(task,&numvar);
  mosek_variables_values_.resize(numvar);
}

int64_t MosekMPCallbackContext::NumExploredNodes() {
  int nnodes;
  MSK_getintinf(task,MSK_IINF_MIO_NUM_SOLVED_NODES,&nnodes);

  return nnodes;
}

MPCallbackEvent MosekMPCallbackContext::Event() {
  return ev;
}

bool MosekMPCallbackContext::CanQueryVariableValues() {
  const MPCallbackEvent where = Event();
  if (where == MPCallbackEvent::kMipSolution) {
    return true;
  }
  return false;
}

double MosekMPCallbackContext::VariableValue(const MPVariable* variable) {
  CHECK(variable != nullptr);
  CHECK(ev == MPCallbackEvent::kMipSolution ||
        ev == MPCallbackEvent::kMipNode)
      << "You can only call VariableValue at "
      << ToString(MPCallbackEvent::kMipSolution) << " or "
      << ToString(MPCallbackEvent::kMipNode)
      << " but called from: " << ToString(ev);
  return mosek_variable_values_[variable->index()];
}

void MosekMPCallbackContext::AddCut(const LinearRange& cutting_plane) { } 
void MosekMPCallbackContext::AddLazyConstraint( const LinearRange& lazy_constraint) { }

double MosekMPCallbackContext::SuggestSolution(const absl::flat_hash_map<const MPVariable*, double>& solution) {
  return 0;
}

struct MPCallbackWithMosekContext {
  MosekMPCallbackContext* context;  
  MPCallback* callback;
  bool * break_solver;
};


int MSKCALL StreamCallbackImpl(
    MSKuserhandle_t h,
    const char * msg) {
  
  MPCallbackWithMosekContext* const callback_with_context = 
      static_cast<MPCallbackWithMosekContext*>(h);

  CHECK(callback_with_context != nullptr);
  CHECK(callback_with_context->context != nullptr);
  callback_with_context->ev = MPCallbackEvent::kMessage;
  callback_with_context->msg = msg;
  
  callback_with_context->callback->RunCallback(callback_with_context->context);

  callback_with_context->msg = nullptr;
}

// NOTE(user): This function must have this exact API, because we are passing
// it to Mosek as a callback.
int MSKCALL CallbackImpl(MSKtask_t task,
                         MSKuserhandle_t h,
                         MSKcallbackcodee where,
                         const double  * dinf,
                         const int     * iinf,
                         const int64_t * liinf) {

  MPCallbackWithMosekContext* const callback_with_context = 
      static_cast<MPCallbackWithMosekContext*>(h);
  CHECK(callback_with_context != nullptr);
  CHECK(callback_with_context->context != nullptr);

  //  kUnknown,
  //  // For regaining control of the main thread in single threaded applications,
  //  // not for interacting with the solver.
  //  kPolling,
  //  // The solver is currently running presolve.
  //  kPresolve,
  //  // The solver is currently running the simplex method.
  //  kSimplex,
  //  // The solver is in the MIP loop (called periodically before starting a new
  //  // node).  Useful to early termination.
  //  kMip,
  //  // Called every time a new MIP incumbent is found.
  //  kMipSolution,
  //  // Called once per pass of the cut loop inside each MIP node.
  //  kMipNode,
  //  // Called in each iterate of IPM/barrier method.
  //  kBarrier,
  //  // The solver is about to log out a message, use this callback to capture it.
  //  kMessage,
  //  // The solver is in multi-objective optimization.
  //  kMultiObj,

  switch (where) {
    // The callback function is called after a new integer solution has been located by the mixed-integer optimizer.
    case MSK_CALLBACK_NEW_INT_MIO:
      MSK_getxx(task,mosek_variable_values_.data());
      callback_with_context->ev = MPCallbackEvent::kMipSolution;
      break;

    case MSK_CALLBACK_BEGIN_DUAL_SIMPLEX:
    case MSK_CALLBACK_BEGIN_DUAL_SIMPLEX_BI:
    case MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX:
    case MSK_CALLBACK_BEGIN_PRIMAL_SIMPLEX_BI:
    case MSK_CALLBACK_END_DUAL_SIMPLEX:
    case MSK_CALLBACK_END_DUAL_SIMPLEX_BI:
    case MSK_CALLBACK_END_PRIMAL_SIMPLEX:
    case MSK_CALLBACK_END_PRIMAL_SIMPLEX_BI:
    case MSK_CALLBACK_IM_PRIMAL_SIMPLEX:
    case MSK_CALLBACK_IM_SIMPLEX:
    case MSK_CALLBACK_IM_SIMPLEX_BI:
    case MSK_CALLBACK_PRIMAL_SIMPLEX:
    case MSK_CALLBACK_UPDATE_DUAL_SIMPLEX:
    case MSK_CALLBACK_UPDATE_DUAL_SIMPLEX_BI:
    case MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX:
    case MSK_CALLBACK_UPDATE_PRIMAL_SIMPLEX_BI:
    case MSK_CALLBACK_UPDATE_SIMPLEX:
      callback_with_context->ev = MPCallbackEvent::kSimplex;
      break;

    case MSK_CALLBACK_INTPNT:
    case MSK_CALLBACK_BEGIN_CONIC:
      callback_with_context->ev = MPCallbackEvent::kBarrier;
      break;

    case MSK_CALLBACK_BEGIN_PRESOLVE:
      callback_with_context->ev = MPCallbackEvent::kPresolve;
      break;

    case MSK_CALLBACK_BEGIN_MIO:
      callback_with_context->ev = MPCallbackEvent::kMip;
      break;

    // The callback function is called from within the basis identification procedure when the primal phase is started.
    case MSK_CALLBACK_BEGIN_PRIMAL_BI:
    // Begin primal feasibility repair.
    case MSK_CALLBACK_BEGIN_PRIMAL_REPAIR:
    // Primal sensitivity analysis is started.
    case MSK_CALLBACK_BEGIN_PRIMAL_SENSITIVITY:
    // The callback function is called when the primal BI setup is started.
    case MSK_CALLBACK_BEGIN_PRIMAL_SETUP_BI:
    // The basis identification procedure has been started.
    case MSK_CALLBACK_BEGIN_BI:
    // The callback function is called from within the basis identification procedure when the dual phase is started.
    case MSK_CALLBACK_BEGIN_DUAL_BI:
    // Dual sensitivity analysis is started.
    case MSK_CALLBACK_BEGIN_DUAL_SENSITIVITY:
    // The callback function is called when the dual BI phase is started.
    case MSK_CALLBACK_BEGIN_DUAL_SETUP_BI:
    // The callback function is called when the infeasibility analyzer is started.
    case MSK_CALLBACK_BEGIN_INFEAS_ANA:
    // The callback function is called when the interior-point optimizer is started.
    case MSK_CALLBACK_BEGIN_INTPNT:
    // Begin waiting for license.
    case MSK_CALLBACK_BEGIN_LICENSE_WAIT:
    // The callback function is called when the optimizer is started.
    case MSK_CALLBACK_BEGIN_OPTIMIZER:
    // Begin QCQO reformulation.
    case MSK_CALLBACK_BEGIN_QCQO_REFORMULATE:
    // The callback function is called when root cut generation is started.
    case MSK_CALLBACK_BEGIN_ROOT_CUTGEN:
    // The callback function is called when the simplex optimizer is started.
    case MSK_CALLBACK_BEGIN_SIMPLEX:
    // The callback function is called from within the basis identification procedure when the simplex clean-up phase is started.
    case MSK_CALLBACK_BEGIN_SIMPLEX_BI:
    // The callback function is called when solution of root relaxation is started.
    case MSK_CALLBACK_BEGIN_SOLVE_ROOT_RELAX:
    // Begin conic reformulation.
    case MSK_CALLBACK_BEGIN_TO_CONIC:
    // The callback function is called from within the conic optimizer after the information database has been updated.
    case MSK_CALLBACK_CONIC:
    // The callback function is called from within the dual simplex optimizer.
    case MSK_CALLBACK_DUAL_SIMPLEX:
    // The callback function is called when the basis identification procedure is terminated.
    case MSK_CALLBACK_END_BI:
    // The callback function is called when the conic optimizer is terminated.
    case MSK_CALLBACK_END_CONIC:
    // The callback function is called from within the basis identification procedure when the dual phase is terminated.
    case MSK_CALLBACK_END_DUAL_BI:
    // Dual sensitivity analysis is terminated.
    case MSK_CALLBACK_END_DUAL_SENSITIVITY:
    // The callback function is called when the dual BI phase is terminated.
    case MSK_CALLBACK_END_DUAL_SETUP_BI:
    // The callback function is called when the infeasibility analyzer is terminated.
    case MSK_CALLBACK_END_INFEAS_ANA:
    // The callback function is called when the interior-point optimizer is terminated.
    case MSK_CALLBACK_END_INTPNT:
    // End waiting for license.
    case MSK_CALLBACK_END_LICENSE_WAIT:
    // The callback function is called when the mixed-integer optimizer is terminated.
    case MSK_CALLBACK_END_MIO:
    // The callback function is called when the optimizer is terminated.
    case MSK_CALLBACK_END_OPTIMIZER:
    // The callback function is called when the presolve is completed.
    case MSK_CALLBACK_END_PRESOLVE:
    // The callback function is called from within the basis identification procedure when the primal phase is terminated.
    case MSK_CALLBACK_END_PRIMAL_BI:
    // End primal feasibility repair.
    case MSK_CALLBACK_END_PRIMAL_REPAIR:
    // Primal sensitivity analysis is terminated.
    case MSK_CALLBACK_END_PRIMAL_SENSITIVITY:
    // The callback function is called when the primal BI setup is terminated.
    case MSK_CALLBACK_END_PRIMAL_SETUP_BI:
    // End QCQO reformulation.
    case MSK_CALLBACK_END_QCQO_REFORMULATE:
    // The callback function is called when root cut generation is terminated.
    case MSK_CALLBACK_END_ROOT_CUTGEN:
    // The callback function is called when the simplex optimizer is terminated.
    case MSK_CALLBACK_END_SIMPLEX:
    // The callback function is called from within the basis identification procedure when the simplex clean-up phase is terminated.
    case MSK_CALLBACK_END_SIMPLEX_BI:
    // The callback function is called when solution of root relaxation is terminated.
    case MSK_CALLBACK_END_SOLVE_ROOT_RELAX:
    // End conic reformulation.
    case MSK_CALLBACK_END_TO_CONIC:
    // The callback function is called from within the basis identification procedure at an intermediate point.
    case MSK_CALLBACK_IM_BI:
    // The callback function is called at an intermediate stage within the conic optimizer where the information database has not been updated.
    case MSK_CALLBACK_IM_CONIC:
    // The callback function is called from within the basis identification procedure at an intermediate point in the dual phase.
    case MSK_CALLBACK_IM_DUAL_BI:
    // The callback function is called at an intermediate stage of the dual sensitivity analysis.
    case MSK_CALLBACK_IM_DUAL_SENSIVITY:
    // The callback function is called at an intermediate point in the dual simplex optimizer.
    case MSK_CALLBACK_IM_DUAL_SIMPLEX:
    // The callback function is called at an intermediate stage within the interior-point optimizer where the information database has not been updated.
    case MSK_CALLBACK_IM_INTPNT:
    // MOSEK is waiting for a license.
    case MSK_CALLBACK_IM_LICENSE_WAIT:
    // The callback function is called from within the LU factorization procedure at an intermediate point.
    case MSK_CALLBACK_IM_LU:
    // The callback function is called at an intermediate point in the mixed-integer optimizer.
    case MSK_CALLBACK_IM_MIO:
    // The callback function is called at an intermediate point in the mixed-integer optimizer while running the dual simplex optimizer.
    case MSK_CALLBACK_IM_MIO_DUAL_SIMPLEX:
    // The callback function is called at an intermediate point in the mixed-integer optimizer while running the interior-point optimizer.
    case MSK_CALLBACK_IM_MIO_INTPNT:
    // The callback function is called at an intermediate point in the mixed-integer optimizer while running the primal simplex optimizer.
    case MSK_CALLBACK_IM_MIO_PRIMAL_SIMPLEX:
    // The callback function is called from within the matrix ordering procedure at an intermediate point.
    case MSK_CALLBACK_IM_ORDER:
    // The callback function is called from within the presolve procedure at an intermediate stage.
    case MSK_CALLBACK_IM_PRESOLVE:
    // The callback function is called from within the basis identification procedure at an intermediate point in the primal phase.
    case MSK_CALLBACK_IM_PRIMAL_BI:
    // The callback function is called at an intermediate stage of the primal sensitivity analysis.
    case MSK_CALLBACK_IM_PRIMAL_SENSIVITY:
    // The callback function is called at an intermediate stage of the conic quadratic reformulation.
    case MSK_CALLBACK_IM_QO_REFORMULATE:
    // The callback is called from within root cut generation at an intermediate stage.
    case MSK_CALLBACK_IM_ROOT_CUTGEN:
    // The callback function is called when the mixed-integer optimizer is restarted. 
    case MSK_CALLBACK_RESTART_MIO:
    // The callback function is called while the task is being solved on a remote server.
    case MSK_CALLBACK_SOLVING_REMOTE:
    // The callback function is called from within the basis identification procedure at an intermediate point in the dual phase.
    case MSK_CALLBACK_UPDATE_DUAL_BI:
    // The callback function is called from within the presolve procedure.
    case MSK_CALLBACK_UPDATE_PRESOLVE:
    // The callback function is called from within the basis identification procedure at an intermediate point in the primal phase.
    case MSK_CALLBACK_UPDATE_PRIMAL_BI:
      callback_with_context->ev = MPCallbackEvent::kPolling;
      break;
    case MSK_CALLBACK_BEGIN_READ:
    case MSK_CALLBACK_BEGIN_WRITE:
    case MSK_CALLBACK_END_READ:
    case MSK_CALLBACK_END_WRITE:
    case MSK_CALLBACK_IM_READ:
    case MSK_CALLBACK_READ_OPF:
    case MSK_CALLBACK_READ_OPF_SECTION:
    case MSK_CALLBACK_WRITE_OPF:
      callback_with_context->ev = MPCallbackEvent::kUnknown;
      break;
  }

  callback_with_context->callback->RunCallback(callback_with_context->context);
  return *(callback_with_context->break_solver) ? 1 : 0;
}

}  // namespace












MPSolutionResponse MosekInterface::DirectlySolveProto(LazyMutableCopy<MPModelRequest> request,
                                      std::atomic<bool>* interrupt) override {
  DCHECK_EQ(interrupt, nullptr);
  const bool log_error = request->enable_internal_solver_output();

  // Here we reuse the Mosek environment to support single-use license that
  // forbids creating a second environment if one already exists.
  return ConvertStatusOrMPSolutionResponse(
      log_error, MosekSolveProto(std::move(request), global_env_));
}

void MosekInterface::CheckedMosekCall(MSKrescodee r) const {
  ::operations_research::CheckedMosekCall(r, task_);
}

// Creates a LP/MIP instance with the specified name and minimization objective.
MosekInterface::MosekInterface(MPSolver* const solver, bool mip)
    : MPSolverInterface(solver),    
      model_(nullptr),
      mip_(mip),
      taskp_(nullptr,MSK_deletetask) {
  CheckedMosekCall(MSK_makeemptytask(nullptr, &task_)); ptask_.reset(&task_);
  CheckedMosekCall(MSK_puttaskname(task,solver_->name.c_str()));
  CheckedMosekCall(MSK_putobjsense(task,maximize_? MSK_OBJECTIVE_SENSE_MAXIMIZE : MSK_OBJECTIVE_SENSE_MINIMIZE));

  CheckedMosekCall(MSK_appendrzerodomain(task_,1,&domidx_rzero));
  CheckedMosekCall(MSK_appendrplusdomain(task_,1,&domidx_rplus));
  CheckedMosekCall(MSK_appendrminusdomain(task_,1,&domidx_rminus));
}





MosekInterface::~MosekInterface() { }


MSKboundkeye MosekInterface::bk_from_bounds(double lb, double ub) {
  return (lb <= ub ? 
            (std::isfinite(lb) ?
              (std::isinfinite(ub) ? 
               (lb < ub ? MSK_BK_RA : MSK_BK_FX) : 
               MSK_BK_LO) :
              (std::isinfinite(ub) ? 
               MSK_BK_UP :
               MSK_BK_FR)) :
           MSK_BK_RA);
}

// ------ Model modifications and extraction -----

void MosekInterface::Reset() {
  const absl::MutexLock lock(&hold_interruptions_mutex_);

  decltype(ptask_) old_taskp(std::move(ptask_));
  MSKtask_t old_task = task_;
  CheckedMosekCall(MSK_makeemptytask(nullptr,&task_));
  ptask_.reset(&task_);

  mp_cons_to_mosek_cons_.clear();

  MosekCloneParameters(task_,old_task);

  ResetExtractionInformation();
}

void MosekInterface::SetOptimizationDirection(bool maximize) {
  InvalidateSolutionSynchronization();
  CheckedMosekCall(MSK_putobjsense(task, maximize ? MSK_OBJECTIVE_SENSE_MAXIMIZE : MSK_OBJECTIVE_SENSE_MINIMIZE);
}

void MosekInterface::SetVariableBounds(int var_index, double lb, double ub) {
  InvalidateSolutionSynchronization();
  MSKboundkey bk = bk_from_bounds(lb,ub);
  CheckedMosekCall(MSK_putvarbound(task,var_index,bk,lb,ub));
}

void MosekInterface::SetVariableInteger(int index, bool integer) {
  InvalidateSolutionSynchronization();

  CheckedMosekCall(MSK_putvartype(task_, index, integer ? MSK_VAR_TYPE_INT : MSK_VAR_TYPE_CONT));
}

void MosekInterface::SetConstraintBounds(int index, double lb, double ub) {
  InvalidateSolutionSynchronization();
  if (mp_cons_to_mosek_cons_[index] >= 0) {
    MSKboundkey bk = bk_from_bounds(lb,ub);
    CheckedMosekCall(MSK_putvarbound(task_, index, bk,bl,bu));
  }
  else {
    int64_t djci = -mp_cons_to_mosek_cons_[index]-1;
    int64_t afei = indcon_afeidx[djci];
    int64_t afeidxs[4] = { afei,afei,afei+1,afei+1 };
    double b[4] = { 0.0,1.0,lb, ub }
    int64_t termsize[2] { 1,3 }
    int64_t domidxs[4] { domidx_rzero, domidx_rzero, domidx_rplus, domidx_rminus };

    if (lb <= ub && lb >= ub) {
      domidxs[2] = domidx_rzero;
      domidxs[3] = domidx_rfree;
    }
    else {
      if (lb < 0 && std::isinfinite(lb)) domidxs[2] = domidx_rfree;
      if (ub > 0 && std::isinfinite(ub)) domidxs[3] = domidx_rfree;
    }

    CheckedMosekCall(MSK_putdjc(task_,djci,4,domidxs,4,afeidxs,b,2,termsize));
  }
}

// Ordinary linear constraint are added as ranged constraints. Indicator
// constraints are added as a disjunctive constraints with constraint lb <= Ax
// <= ub here K is a value, a range or a half-open range, and X is a binary
// variable as 
// (X < 0.5) OR (lb < Ax AND Ax < ub)
//
void MosekInterface::AddRowConstraint(MPConstraint* const ct) {
  int conidx;
  CheckedMosekCall(MSK_getnumcon(task_,&conidx));
  CheckedMosekCall(MSK_appendcons(task_,1));
  mp_cons_to_mosek_cons_.push(conidx);

  double lb = ct->lb(); 
  double ub = ct->ub(); 

  MSKboundkey bk = bk_from_bounds(lb,ub);
  CheckedMosekCall(MSK_putconbound(task_,bk,lb,ub));
  std::vector<double> cof; cof.reserve(ct->coefficients_.size());
  std::vector<int> subj; subj.reserve(cof.size());
  for (auto it : ct->terms()) {
    subj.push_back(it->first);
    cof.push_back(it->second);
  }

  CheckedMosekCall(MSK_putarow(task_,conidx,subj.size(),subj.data(),cof.data()));
}

bool MosekInterface::AddIndicatorConstraint(MPConstraint* const ct) {
  int64_t djci,afei;
  CheckedMosekCall(MSK_getnumdjc(task_,&djci));
  CheckedMosekCall(MSK_appenddjcs(task_,1));
  CheckedMosekCall(MSK_getnumafe(task_,&afei));
  CheckedMosekCall(MSK_appendafes(task_,2));
  mp_cons_to_mosek_cons_.push_back(-1-djci);
  indcon_afeidx.push_back(afei);
  
  int indvar = ct->indicator_variable();

  CheckedMosekCall(MSK_putvartype(task_,indvar,MSK_VAR_TYPE_INT));

  // TODO: Check if variable type and bounds for an indicator variable are set by the interface.
  CheckedMosekCall(MSK_putvarbound(task,indvar,MSK_BK_RA,0.0,1.0));
  
  {
    double lb = ct->lb(); 
    double ub = ct->ub(); 

    int64_t domidxs[4] = { domidx_rzero,domidx_rzero, domidx_rplus, domidx_rminus };
    int64_t afeidxs[4] = { afei, afei, afei+1, afei+1 };
    double  b[3]       = { 0.0, 1.0, lb,ub };
    int64_t termsize[2] = {1, 3};

    if (lb <= ub && lb >= ub) {
      domidxs[2] = domidx_rzero;
      domidxs[3] = domidx_rfree;
    }
    else {
      if (lb < 0 && std::isinfinite(lb)) domidxs[2] = domidx_rfree;
      if (ub > 0 && std::isinfinite(ub)) domidxs[3] = domidx_rfree;    
    }

    CheckedMosekCall(MSK_putdjc(task_, djci, 4,domidxs,4,afeidxs,b,2,termsize));
  }
  {
    std::vector<double> cof; cof.reserve(ct->coefficients_.size());
    std::vector<int> subj; subj.reserve(cof.size());
    for (auto it : ct->terms()) {
      subj.push_back(it->first);
      cof.push_back(it->second);
    }
    CheckedMosekCall(MSK_putafefrow(task_,afei+1,subj.size(),subj.data(),cof.data()));
  }
  {
    double c = 1.0;
    CheckedMosekCall(MSK_putafefrow(task_,afei,1,&indvar,&c));
  }

  return true;
}

void MosekInterface::AddVariable(MPVariable* const var) {
  int j;
  CheckedMosekCall(MSK_getnumvar(task_));
  CheckedMosekCall(MSK_appendvars(task_,1));
  double lb = ct->lb(); 
  double ub = ct->ub(); 

  MSKboundkey bk = bk_from_bounds(lb,ub);
  checkedmosekcall(msk_putvarbound(task_,j,bk,lb,ub));
  if (ct->integer())
    checkedmosekcall(msk_putvartype(task_,j,msk_var_type_int));
}

void MosekInterface::SetCoefficient(MPConstraint* const constraint,
                                     const MPVariable* const variable,
                                     double new_value, double old_value) {
  InvalidateSolutionSynchronization();
  
  ssize_t coni = mp_cons_to_mosek_cons_[constraint->index()];
  if (coni >= 0) {
    CheckedMosekCall(MSK_putaij(task_, coni, variable->index(), new_value));
  }
  else {
    int64_t djci = -coni-1;
    int64_t djcnumafe;
    int64_t afeidxs[3];
    CheckedMosekCall(MSK_getdjcnumafe(task_,djci,&djcnumafe));
    CHECK_OK(djcnumafe,3) << "Invalid internal constraint data";
    CheckedMosekCall(MSK_getdjcafeidxlist(task_,djci,&afeidx));

    CheckedMosekCall(MSK_putafefentry(task_,afeidxs[1],variable->index(),new_value));
  }
}

// Question: Is an indicator constraint ever cleared? What exactly does that mean? 
void MosekInterface::ClearConstraint(MPConstraint* const constraint) {
  InvalidateSolutionSynchronization();
  // TODO: Cleanup if afe nonzeros etc?
  ssize_t coni = mp_cons_to_mosek_cons_[constraint->index()];
  if (coni >= 0) {
    CheckedMosekCall(MSK_putarow(task_,coni,0,nullptr,nullptr));
    CheckedMosekCall(MSK_putconbound(task_,coni,MSK_BK_FR,0.0,0.0));
  }
  else {
    int64_t djci = -coni-1;
    CheckedMosekCall(MSK_putdjc(task_,djci,0,nullptr,0,nullptr,nullptr,0,nullptr));
  }
}

void MosekInterface::SetObjectiveCoefficient(const MPVariable* const variable,
                                              double coefficient) {
  InvalidateSolutionSynchronization();
  CheckedMosekCall(MSK_putcj(task_, variable->index(),coefficient));
}

void MosekInterface::SetObjectiveOffset(double value) {
  InvalidateSolutionSynchronization();

  CheckedMosekCall(MSK_putcfix(task_,value));
}

void MosekInterface::ClearObjective() {
  InvalidateSolutionSynchronization();
  int numvar;
  CheckedMosekCall(MSK_getnumvar(task_,&numvar));
  for (int i = 0; i < numvar; ++i)
    MSK_putcj(task_,i,0.0);
  MSK_putcfix(task_,0.0);
}

// ------ Query statistics on the solution and the solve ------

int64_t MosekInterface::iterations() const {
  if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfIterations;
  // TODO: Iter Count
  int32_t psim_iter,dsim_iter,intpnt_iter;
  CheckedMosekCall(MSK_getnaintinf(task_,"MSK_IINF_SIM_DUAL_ITER",&psim_iter));
  CheckedMosekCall(MSK_getnaintinf(task_,"MSK_IINF_SIM_PRIMAL_ITER",&dsim_iter));
  CheckedMosekCall(MSK_getnaintinf(task_,"MSK_IINF_INTPNT_ITER",&intpnt_iter));

  return intpnt_iter > 0 ? intpnt_iter : psim_iter+dsim_iter;
}

int64_t MosekInterface::nodes() const {
  if (mip_) {
    if (!CheckSolutionIsSynchronized()) return kUnknownNumberOfNodes;
    int nnodes;
    CheckedMosekCall(MSK_getnaintinf(task_,"MSK_IINF_MIO_NUM_SOLVED_NODES",&nnodes));
    return nnodes;
  } else {
    LOG(DFATAL) << "Number of nodes only available for discrete problems.";
    return kUnknownNumberOfNodes;
  }
}

// Returns the basis status of a row.
MPSolver::BasisStatus MosekInterface::row_status(int constraint_index) const {
  auto coni = mp_cons_to_mosek_cons_[constraint_index];
  if (coni < 0) {
    log(dfatal) << "Basis status only available for continuous problems.";
  }

  int soldef;

  CheckedMosekCall(MSK_solutiondef(task_,MSK_SOL_BAS,soldef));
  if (! soldef) {
    log(dfatal) << "Basis status only available when a basis solution has been found.";
    return MPSolver::FREE;
  }

  MSKstakeye sk;
  CheckedMosekCall(MSK_getskcslice(task_,coni,sk));
  
  switch sk {
    case MSK_SK_BAS: return MPSolver::BASIS;
    case MSK_SK_LO: return MPSolver::AT_LOWER_BOUND;
    case MSK_SK_UP: return MPSolver::AT_UPPER_BOUND;
    case MSK_SK_FX: return MPSolver::FIXED_VALUE;
    case MSK_SK_FR: return MPSolver::FREE;

    default:
      log(dfatal) << "Basis status only available when a basis solution has been found.";
      return MPSolver::FREE;
  }
}

// Returns the basis status of a column.
MPSolver::BasisStatus MosekInterface::column_status(int variable_index) const {
  int soldef;

  CheckedMosekCall(MSK_solutiondef(task_,MSK_SOL_BAS,soldef));
  if (! soldef) {
    log(dfatal) << "Basis status only available when a basis solution has been found.";
    return MPSolver::FREE;
  }

  MSKstakeye sk;
  CheckedMosekCall(MSK_getskxslice(task_,variable_index,sk));
  
  switch sk {
    case MSK_SK_BAS: return MPSolver::BASIS;
    case MSK_SK_LO: return MPSolver::AT_LOWER_BOUND;
    case MSK_SK_UP: return MPSolver::AT_UPPER_BOUND;
    case MSK_SK_FX: return MPSolver::FIXED_VALUE;
    case MSK_SK_FR: return MPSolver::FREE;

    default:
      log(dfatal) << "Basis status only available when a basis solution has been found.";
      return MPSolver::FREE;
  }
}

// Extracts new variables.
void MosekInterface::ExtractNewVariables() {
  int numvar;
  auto total_num_vars = solver_->variables_().size();
  CheckedMosekCall(MSK_getnumvar(task_,&numvar));
  if (total_num_vars > numvar) {
    CheckedMosekCall(MSK_appendvars(task_,total_num_vars-numvar));
    auto obj = solver_->Objective();

    for (int j = numvar; numvar < total_num_vars; ++i) {
      auto var = solver_->variables_()[j];
      set_variable_as_extracted(j, true);
      MSKboundkey bk = bk_from_bounds(var->lb(),var->ub());
      CheckedMosekCall(MSK_putvarbound(task_,j,bk,var->lb(),var->ub()));
      if (var->integer()) {
        CheckedMosekCall(MSK_putvartype(task_,j,MSK_VAR_TYPE_INT));
      }

      double cj = obj->GetCoefficient(var);
      if (cj > 0 || cj < 0) {
        CheckedMosekCall(MSK_putcj(task_,j,cj));
      }


      for (int i = 0; i < mp_cons_to_mosek_cons_.size(); ++i) {
        auto coni = mp_cons_to_mosek_cons_[i];
        const MPConstraint* ct = solver_->constraints()[i];
        if (coni >= 0) {
          for (const auto &item : ct->coefficients_) {
            if (item->first >= numvar) {
              CheckedMosekCall(MSK_putaij(task_,coni,item->first,item->second()));
            }
          }
        }
        else {
          auto djci = -coni-1;
          auto afei = indcon_afeidx[djci];
          for (const auto &item : ct->coefficients_) {
            if (item->first >= numvar) {
              CheckedMosekCall(MSK_putafefentry(task_,afei+1,item->first,item->second));
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
  CheckedMosekCall(MSK_putobjsense(task,maximize_ ? MSK_OBJECTIVE_SENSE_MAXIMIZE : MSK_OBJECTIVE_SENSE_MINIMIZE));
  auto obj = solver->Objective();
  CheckedMosekCall(MSK_putcfix(task_,obj->offset()));
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
  std::string key,value;
  while (data.size() > 0) {
    auto p = data.find('\n');
    if (p == std::string_view::npos) p = data.size();
    std::string_view line = data.substr(0,p);
    
    auto eq_pos = line.find('=');
    if (eq_pos != std::string_view::npos) {
      key.clear();
      value.clear();
      key += line.substr(0,eq_pos);
      value += line.substr(eq_pos+1);

      if (MSK_RES_OK != MSK_putparam(task,key.c_str(),value.c_str())) {
        LOG(WARNING) << "Failed to set parameters '" << key << "' to '" << value << "'";
      }
    }

    data = data.substr(p+1);
  }
  return true;
}

void MosekInterface::SetRelativeMipGap(double value) {
  CheckedMosekCall(MSK_putdouparam(task_,MSK_DPAR_MIO_REL_GAP_CONST,value));
}

void MosekInterface::SetPrimalTolerance(double value) {
  CheckedMosekCall(MSK_putdparam(task_, MSK_DPAR_INTPNT_TOL_PFEAS));
  CheckedMosekCall(MSK_putdparam(task_, MSK_DPAR_BASIS_TOL_X));
}

void MosekInterface::SetDualTolerance(double value) {
  CheckedMosekCall(MSK_putdparam(task_, MSK_DPAR_INTPNT_TOL_DFEAS));
  CheckedMosekCall(MSK_putdparam(task_, MSK_DPAR_BASIS_TOL_S));
}


void MosekInterface::SetPresolveMode(int value) {
  switch (value) {
    case MPSolverParameters::PRESOLVE_OFF: {
      CheckedMosekCall( MSK_putintparam(task_,MSK_IPAR_PRESOLVE_USE,MSK_OFF));
      break;
    }
    case MPSolverParameters::PRESOLVE_ON: {
      CheckedMosekCall( MSK_putintparam(task_,MSK_IPAR_PRESOLVE_USE,MSK_ON));
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
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_INTPNT_SCALING,MSK_SCALING_NONE));
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_SIM_SCALING,MSK_SCALING_NONE));
      break;
    case MPSolverParameters::SCALING_ON:
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_INTPNT_SCALING,MSK_SCALING_FREE));
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_SIM_SCALING,MSK_SCALING_FREE));
      break;
    default:
      // Leave the parameters untouched.
      break;
  }
}

void MosekInterface::SetLpAlgorithm(int value) {
  switch (value) {
    case MPSolverParameters::DUAL:
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_DUAL_SIMPLEX))
      break;
    case MPSolverParameters::PRIMAL:
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_PRIMAL_SIMPLEX))
      break;
    case MPSolverParameters::BARRIER:
      CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT));
      break;
    default:
      SetIntegerParamToUnsupportedValue(MPSolverParameters::LP_ALGORITHM,
                                        value);
  }
}

int MosekInterface::SolutionCount() const {
  int soldef;
  MSK_solutiondef(task_,MSK_SOL_ITG,&soldef); if (soldef) return 1;
  MSK_solutiondef(task_,MSK_SOL_BAS,&soldef); if (soldef) return 1;
  MSK_solutiondef(task_,MSK_SOL_ITR,&soldef); if (soldef) return 1;
  return 0;
}

MPSolver::ResultStatus MosekInterface::Solve(const MPSolverParameters& param) {
  WallTimer timer;
  timer.Start();

  // Set log level.
  CheckedMosekCall(MSK_putintparam(task_,MSK_IPAR_LOG, quiet ? 0 : 10));

  ExtractModel();
  // Sync solver.
  VLOG(1) << absl::StrFormat("Model built in %s.",
                             absl::FormatDuration(timer.GetDuration()));

  int numvar; MSK_getnumvar(task_, &numvar);
  // Set solution hints. Currently this only affects integer solution.
  if (solver_->solution_hint_.size() > 0) {
    std::vector<double> xx(numvar);
    for (const std::pair<const MPVariable*, double>& p : solver_->solution_hint_) {
      xx[p.first->index()] = p.second;
    }
    MSK_putxx(task_,MSK_SOL_ITG, xx.data());
  }

  // Time limit.
  if (solver_->time_limit() != 0) {
    VLOG(1) << "Setting time limit = " << solver_->time_limit() << " ms.";
    CheckedMosekCall(MSK_putdouparam(task_, MSK_DPAR_OPTIMIZER_MAX_TIME,solver_->time_limit_in_secs()));
  }

  // We first set our internal MPSolverParameters from 'param' and then set
  // any user-specified internal solver parameters via
  // solver_specific_parameter_string_.
  // Default MPSolverParameters can override custom parameters (for example for
  // presolving) and therefore we apply MPSolverParameters first.
  SetParameters(param);
  solver_->SetSolverSpecificParametersAsString(
      solver_->solver_specific_parameter_string_);

  std::unique_ptr<MosekMPCallbackContext> mosek_context;
  MPCallbackWithMosekContext mp_callback_with_context = { .break_solver : &break_solver_, .callback : callback, .context : nullptr };
  if (callback_ == nullptr) {
    CheckedMosekCall(MSK_putcallbackfunc(task_, nullptr, nullptr));
  } else {
    mosek_context = std::make_unique<MosekMPCallbackContext>(task_);
    mp_callback_with_context.context = mosek_context.get();
  }

  // remove any pre-existing solution in task that are not relevant for the result.
  MSK_putintparam(task_,MSK_IPAR_REMOVE_UNUSED_SOLUTIONS,MSK_OK);

  // Solve
  timer.Restart();

  MSKrescodee trm;
  {
    MSK_putcallbackfunc(task_,CallbackImpl,&mp_callback_with_context);
    MSK_linkfunctotaskstream(task_,MSK_STREAM_LOG,&mp_callback_with_context,StreamCallbackImpl);
    CheckedMosekCall(MSK_optimizetrm(task,&trm));
    MSK_linkfunctotaskstream(task_,MSK_STREAM_LOG,nullptr,nullptr);
    MSK_putcallbackfunc(task_,nullptr,nullptr);
  }

  VLOG(1) << absl::StrFormat("Solved in %s.",
                             absl::FormatDuration(timer.GetDuration()));
  // Get the status.
  MSKprostae prosta = MSK_SOL_PRO_UNKNOWN;
  MSKsolstae solsta = MSK_SOL_STA_UNKNOWN;
  MSKsoltypee whichsol;
  bool soldefined = true;
  {
    int soldef;
    whichsol = MSK_SOL_ITG; MSK_solutiondef(task_,whichsol,&soldef); 
    if (! soldef) {
      whichsol = MSK_SOL_BAS; MSK_solutiondef(task_,whichsol,&soldef); 
    }
    if (! soldef) {
      whichsol = MSK_SOL_ITR; MSK_solutiondef(task_,whichsol,&soldef); 
    }
    soldefined = soldef != 0;
  }

  if (soldefined) {
    MSK_getprosta(task_, whichsol, &prosta);
    MSK_getsolsta(task_, whichsol, &solsta);
  }

  VLOG(1) << absl::StrFormat("Solution status %d.\n", prosta);
  const int solution_count = SolutionCount();

  if (solsta == MSK_SOL_STA_OPTIMAL ||
      solsta == MSK_SOL_STA_INTEGER_OPTIMAL) {
      result_status = MPSolver::OPTIMAL;
  }
  else if (prosta == MSK_SOL_STA_PRIM_AND_DUAL_FEAS) {
      result_status_ = MPSolver::FEASIBLE;
  }
  else if (prosta == MSK_PRO_STA_PRIM_INFEAS) {
      result_status_ = MPSolver::INFEASIBLE;
  }
  else if (prosta == MSK_PRO_STA_DUAL_INFEAS) {
      result_status_ = MPSolver::UNBOUNDED;
  }
  else if (prosta == MSK_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED)
      // TODO(user): We could introduce our own "infeasible or
      // unbounded" status.
      result_status_ = MPSolver::INFEASIBLE;
  }
  else {
      result_status_ = MPSolver::NOT_SOLVED;
  }

  // Get best objective bound value
  if (IsMIP() && (result_status_ == MPSolver::FEASIBLE &&
                  result_status_ == MPSolver::OPTIMAL)) {
    MSK_getdouinf(task_,MSK_DINF_MIO_OBJ_BOUND,&best_objective_bound_);
    VLOG(1) << "best bound = " << best_objective_bound_;
  }

  if (solution_count > 0 && (result_status_ == MPSolver::FEASIBLE ||
                             result_status_ == MPSolver::OPTIMAL)) {
    current_solution_index_ = 0;
    // Get the results.
    MSK_getprimalobj(task_,whichsol,&objective_value_);
    VLOG(1) << "objective = " << objective_value_;


    std::vector<double> xx(numvar);
    CheckedMosekCall(MSK_getxx(task_,whichsol,xx.data()));
    {
      for (int i = 0; i < solver_->variables_.size(); ++i) {
        MPVariable* const var = solver_->variables_[i];
        var->set_solution_value(xx[i]);
        VLOG(3) << var->name() << ", value = " << val;
      }
    }
    if (!mip_) {
      {
        std::vector<double> slx(numvar);
        std::vector<double> sux(numvar);
        
        CheckedMosekCall(MSK_getslx(task_,whichsol,slx.data()));
        CheckedMosekCall(MSK_getsux(task_,whichsol,sux.data()));

        for (int i = 0; i < solver_->variables_.size(); ++i) {
          MPVariable* const var = solver_->variables_[i];
          var->set_reduced_cost(slx[i]-sux[i]);
          VLOG(4) << var->name() << ", reduced cost = " << (slx[i]-sux[i]);
        }
      }

      {
        size_t numcon = mp_cons_to_mosek_cons_.size();
        std::vector<double> y(numcon);

        CheckedMosekCall(MSK_gety(task_,whichsol,y.data()));

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
bool MosekInterface::NextSolution() {
  return false;
}

void MosekInterface::Write(const std::string& filename) {
  //if (sync_status_ == MUST_RELOAD) {
  //  Reset();
  //}
  //ExtractModel();
  VLOG(1) << "Writing Mosek Task file \"" << filename << "\".";
  MSKrescodee r = MSK_writedata(task,filename.c_str());
  if (MSK_RES_OK != r) {
    auto lasterr = MosekLastError(task_);
    LOG(WARNING) << "Failed to write Task. Error (" << lasterr.second << "): " << lasterr.first; 
  }
}

MPSolverInterface* BuildMosekInterface(bool mip, MPSolver* const solver) {
  return new MosekInterface(solver, mip);
}

void MosekInterface::SetCallback(MPCallback* mp_callback) {
  callback_ = mp_callback;
}

}  // namespace operations_research


#endif // USE_MOSEK
