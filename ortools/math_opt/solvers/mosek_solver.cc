// Copyright 2010-2024 Google LLC
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


#include "ortools/math_opt/solvers/mosek_solver.h"
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "mosek.h"
#include "mosek/mosekwrp.h"
#include "absl/algorithm/container.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "ortools/base/protoutil.h"
#include "ortools/base/status_builder.h"
#include "ortools/base/status_macros.h"
#include "ortools/math_opt/core/empty_bounds.h"
#include "ortools/math_opt/core/inverted_bounds.h"
#include "ortools/math_opt/core/math_opt_proto_utils.h"
#include "ortools/math_opt/core/solver_interface.h"
#include "ortools/math_opt/core/sorted.h"
#include "ortools/math_opt/core/sparse_vector_view.h"
#include "ortools/math_opt/parameters.pb.h"
#include "ortools/math_opt/result.pb.h"
#include "ortools/math_opt/solution.pb.h"
#include "ortools/math_opt/solvers/mosek.pb.h"
#include "ortools/math_opt/solvers/message_callback_data.h"
#include "ortools/util/solve_interrupter.h"
#include "ortools/util/status_macros.h"

namespace operations_research::math_opt {
namespace {

//constexpr absl::string_view kOutputFlag = "output_flag";
//constexpr absl::string_view kLogToConsole = "log_to_console";

//constexpr SupportedProblemStructures kHighsSupportedStructures = {
//    .integer_variables = SupportType::kSupported,
//    .quadratic_objectives = SupportType::kNotImplemented};

#if 0
absl::Status ToStatus(const HighsStatus status) {
  switch (status) {
    case HighsStatus::kOk:
      return absl::OkStatus();
    case HighsStatus::kWarning:
      // There doesn't seem to be much we can do with this beyond ignoring it,
      // which does not seem best. Highs returns a warning when you solve and
      // don't get a primal feasible solution, but MathOpt does not consider
      // this to be warning worthy.
      return absl::OkStatus();
    case HighsStatus::kError:
      return util::InternalErrorBuilder() << "HighsStatus: kError";
    default:
      return util::InternalErrorBuilder()
             << "unexpected HighsStatus: " << static_cast<int>(status);
  }
}

absl::Status ToStatus(const OptionStatus option_status) {
  switch (option_status) {
    case OptionStatus::kOk:
      return absl::OkStatus();
    case OptionStatus::kUnknownOption:
      return absl::InvalidArgumentError("option name was unknown");
    case OptionStatus::kIllegalValue:
      // NOTE: highs returns this if the option type is wrong or if the value
      // is out of bounds for the option.
      return absl::InvalidArgumentError("option value not valid for name");
  }
  return util::InternalErrorBuilder()
         << "unexpected option_status: " << static_cast<int>(option_status);
}

absl::StatusOr<int> SafeIntCast(const int64_t i, const absl::string_view name) {
  if constexpr (sizeof(int) >= sizeof(int64_t)) {
    return static_cast<int>(i);
  } else {
    const int64_t kMin = static_cast<int64_t>(std::numeric_limits<int>::min());
    const int64_t kMax = static_cast<int64_t>(std::numeric_limits<int>::max());
    if (i < kMin || i > kMax) {
      return util::InvalidArgumentErrorBuilder()
             << name << " has value " << i
             << " not representable as an int (the range [" << kMin << ", "
             << kMax << "]) and thus is not supported for HiGHS";
    }
    return static_cast<int>(i);
  }
}

template <typename T>
int64_t CastInt64StaticAssert(const T value) {
  static_assert(std::is_integral_v<T>);
  static_assert(sizeof(T) <= sizeof(int64_t));
  return static_cast<int64_t>(value);
}

// Note: the highs solver has very little documentation, but you can find some
// here https://www.gams.com/latest/docs/S_HIGHS.html.
absl::StatusOr<std::unique_ptr<HighsOptions>> MakeOptions(
    const SolveParametersProto& parameters, const bool has_log_callback,
    const bool is_integer) {
  // Copy/move seem to be broken for HighsOptions, asan errors.
  auto result = std::make_unique<HighsOptions>();

  if (parameters.highs().bool_options().contains(kOutputFlag)) {
    result->output_flag = parameters.highs().bool_options().at(kOutputFlag);
  } else {
    result->output_flag = parameters.enable_output() || has_log_callback;
  }
  // This feature of highs is pretty confusing/surprising. To use a callback,
  // you need log_to_console to be true. From this line:
  //   https://github.com/ERGO-Code/HiGHS/blob/master/src/io/HighsIO.cpp#L101
  // we see that if log_to_console is false and log_file_stream are null, we get
  // no logging at all.
  //
  // Further, when the callback is set, we won't log to console anyway. But from
  // the names it seems like it should be
  // result.log_to_console = parameters.enable_output() && !has_log_callback;
  if (parameters.highs().bool_options().contains(kLogToConsole)) {
    result->log_to_console =
        parameters.highs().bool_options().at(kLogToConsole);
  } else {
    result->log_to_console = result->output_flag;
  }
  if (parameters.has_time_limit()) {
    OR_ASSIGN_OR_RETURN3(
        const absl::Duration time_limit,
        util_time::DecodeGoogleApiProto(parameters.time_limit()),
        _ << "invalid time_limit value for HiGHS.");
    result->time_limit = absl::ToDoubleSeconds(time_limit);
  }
  if (parameters.has_iteration_limit()) {
    if (is_integer) {
      return util::InvalidArgumentErrorBuilder()
             << "iteration_limit not supported for HiGHS on problems with "
                "integer variables";
    }
    ASSIGN_OR_RETURN(
        const int iter_limit,
        SafeIntCast(parameters.iteration_limit(), "iteration_limit"));

    result->simplex_iteration_limit = iter_limit;
    result->ipm_iteration_limit = iter_limit;
  }
  if (parameters.has_node_limit()) {
    ASSIGN_OR_RETURN(result->mip_max_nodes,
                     SafeIntCast(parameters.node_limit(), "node_limit"));
  }
  if (parameters.has_cutoff_limit()) {
    // TODO(b/271606858) : It may be possible to get this working for IPs via
    //  objective_bound. For LPs this approach will not work.
    return absl::InvalidArgumentError("cutoff_limit not supported for HiGHS");
  }
  if (parameters.has_objective_limit()) {
    if (is_integer) {
      return util::InvalidArgumentErrorBuilder()
             << "objective_limit not supported for HiGHS solver on integer "
                "problems.";
    } else {
      // TODO(b/271616762): it appears that HiGHS intended to support this case
      // but that it is just broken, we should set result.objective_target.
      return absl::InvalidArgumentError(
          "objective_limit for LP appears to have a missing/broken HiGHS "
          "implementation, see b/271616762");
    }
  }
  if (parameters.has_best_bound_limit()) {
    if (is_integer) {
      return util::InvalidArgumentErrorBuilder()
             << "best_bound_limit not supported for HiGHS solver on integer "
                "problems.";
    } else {
      result->objective_bound = parameters.best_bound_limit();
    }
  }
  if (parameters.has_solution_limit()) {
    result->mip_max_improving_sols = parameters.solution_limit();
  }
  if (parameters.has_threads()) {
    // Do not assign result.threads = parameters.threads() here, this is
    // requires global synchronization. See
    // cs/highs/src/lp_data/Highs.cpp:607
    return util::InvalidArgumentErrorBuilder()
           << "threads not supported for HiGHS solver, this must be set using "
              "globals, see HiGHS documentation";
  }
  if (parameters.has_random_seed()) {
    result->random_seed = parameters.random_seed();
  }
  if (parameters.has_absolute_gap_tolerance()) {
    result->mip_abs_gap = parameters.absolute_gap_tolerance();
  }
  if (parameters.has_relative_gap_tolerance()) {
    result->mip_rel_gap = parameters.relative_gap_tolerance();
  }
  if (parameters.has_solution_pool_size()) {
    return util::InvalidArgumentErrorBuilder()
           << "solution_pool_size not supported for HiGHS";
  }
  if (parameters.lp_algorithm() != LP_ALGORITHM_UNSPECIFIED) {
    if (is_integer) {
      return util::InvalidArgumentErrorBuilder()
             << "lp_algorithm is not supported for HiGHS on problems with "
                "integer variables";
    }
    switch (parameters.lp_algorithm()) {
      case LP_ALGORITHM_PRIMAL_SIMPLEX:
        result->solver = ::kSimplexString;
        result->simplex_strategy = ::kSimplexStrategyPrimal;
        break;
      case LP_ALGORITHM_DUAL_SIMPLEX:
        result->solver = ::kSimplexString;
        result->simplex_strategy = ::kSimplexStrategyDual;
        break;
      case LP_ALGORITHM_BARRIER:
        result->solver = ::kIpmString;
        break;
      default:
        return util::InvalidArgumentErrorBuilder()
               << "unsupported lp_algorithm: "
               << LPAlgorithmProto_Name(parameters.lp_algorithm());
    }
  }
  if (parameters.presolve() != EMPHASIS_UNSPECIFIED) {
    if (parameters.presolve() == EMPHASIS_OFF) {
      result->presolve = ::kHighsOffString;
    } else {
      result->presolve = ::kHighsOnString;
    }
  }
  if (parameters.cuts() != EMPHASIS_UNSPECIFIED) {
    return util::InvalidArgumentErrorBuilder()
           << "cuts solve parameter unsupported for HiGHS";
  }
  if (parameters.heuristics() != EMPHASIS_UNSPECIFIED) {
    switch (parameters.heuristics()) {
      case EMPHASIS_OFF:
        result->mip_heuristic_effort = 0.0;
        break;
      case EMPHASIS_LOW:
        result->mip_heuristic_effort = 0.025;
        break;
      case EMPHASIS_MEDIUM:
        result->mip_heuristic_effort = 0.05;
        break;
      case EMPHASIS_HIGH:
        result->mip_heuristic_effort = 0.1;
        break;
      case EMPHASIS_VERY_HIGH:
        result->mip_heuristic_effort = 0.2;
        break;
      default:
        return util::InvalidArgumentErrorBuilder()
               << "unexpected value for solve_parameters.heuristics of: "
               << parameters.heuristics();
    }
  }
  if (parameters.scaling() != EMPHASIS_UNSPECIFIED) {
    // Maybe we can do better here? Not clear how highs scaling works
    if (parameters.scaling() == EMPHASIS_OFF) {
      result->simplex_scale_strategy = ::kSimplexScaleStrategyOff;
    }
  }
  for (const auto& [name, value] : parameters.highs().string_options()) {
    if (name == kOutputFlag || name == kLogToConsole) {
      // This case was handled specially above. We need to do the output
      // parameters first, as we don't want extra logging while setting options.
      continue;
    }
    RETURN_IF_ERROR(ToStatus(setLocalOptionValue(result->log_options, name,
                                                 result->log_options,
                                                 result->records, value)))
        << "error setting string option name: " << name
        << " to value:" << value;
  }
  for (const auto& [name, value] : parameters.highs().double_options()) {
    RETURN_IF_ERROR(ToStatus(
        setLocalOptionValue(result->log_options, name, result->records, value)))
        << "error setting double option name: " << name
        << " to value:" << value;
  }
  for (const auto& [name, value] : parameters.highs().int_options()) {
    RETURN_IF_ERROR(ToStatus(
        setLocalOptionValue(result->log_options, name, result->records, value)))
        << "error setting int option name: " << name << " to value:" << value;
  }
  for (const auto& [name, value] : parameters.highs().bool_options()) {
    RETURN_IF_ERROR(ToStatus(
        setLocalOptionValue(result->log_options, name, result->records, value)))
        << "error setting bool option name: " << name << " to value:" << value;
  }
  return result;
}

double DualObjective(const HighsInfo& highs_info, const bool is_integer) {
  // TODO(b/290359402): for is_integer = false, consider computing the objective
  // of a returned dual feasible solution instead.
  return is_integer ? highs_info.mip_dual_bound
                    : highs_info.objective_function_value;
}
// Note that this is the expected/required function signature for highs logging
// callbacks as set with Highs::setLogCallback().
void HighsLogCallback(HighsLogType, const char* const message,
                      void* const log_callback_data) {
  BufferedMessageCallback& buffered_callback =
      *static_cast<BufferedMessageCallback*>(log_callback_data);
  buffered_callback.OnMessage(message);
}

// highs_info must be valid. Does not fill in solve time.
absl::StatusOr<SolveStatsProto> ToSolveStats(const HighsInfo& highs_info) {
  SolveStatsProto result;
  // HiGHS does to not report simplex and barrier count for mip. There is no
  // way to extract it, as it is held in
  // HighsMipSolver.mipdata_.total_lp_iterations, but the HighsMipSolver
  // object is created and destroyed within a single call to Highs.run() here:
  // https://github.com/ERGO-Code/HiGHS/blob/master/src/lp_data/Highs.cpp#L2976
  result.set_simplex_iterations(std::max(
      int64_t{0}, CastInt64StaticAssert(highs_info.simplex_iteration_count)));
  result.set_barrier_iterations(std::max(
      int64_t{0}, CastInt64StaticAssert(highs_info.ipm_iteration_count)));
  result.set_node_count(std::max(int64_t{0}, highs_info.mip_node_count));
  return result;
}

// Returns nullopt for nonbasic variables when the upper/lower status is not
// known.
absl::StatusOr<std::optional<BasisStatusProto>> ToBasisStatus(
    const HighsBasisStatus highs_basis, const double lb, const double ub,
    const std::optional<double> value) {
  switch (highs_basis) {
    case HighsBasisStatus::kBasic:
      return BASIS_STATUS_BASIC;
    case HighsBasisStatus::kUpper:
      return BASIS_STATUS_AT_UPPER_BOUND;
    case HighsBasisStatus::kLower:
      // Note: highs returns lower for fixed.
      // https://github.com/ERGO-Code/HiGHS/blob/master/src/lp_data/HConst.h#L192
      // TODO(b/272767311): investigate returning fixed instead.
      return BASIS_STATUS_AT_LOWER_BOUND;
    case HighsBasisStatus::kZero:
      return BASIS_STATUS_FREE;
    // TODO(b/272767311): this can potentially be simplified/deleted, we need
    // to see if HiGHS will ever return kNonbasic/decide if we want to support
    // kNonbasic as part of the mathopt starting basis API.
    case HighsBasisStatus::kNonbasic: {
      const bool lb_finite = std::isfinite(lb);
      const bool ub_finite = std::isfinite(ub);
      // TODO(b/272767311): it would be better if this was configurable, use a
      // small/conservative value for now (if it fails, we fail to return a
      // basis).
      constexpr double kAtBoundTolerance = 1.0e-10;
      if (lb_finite && ub_finite) {
        if (lb == ub) {
          return BASIS_STATUS_FIXED_VALUE;
        } else if (value.has_value() &&
                   std::abs(lb - *value) < kAtBoundTolerance) {
          return BASIS_STATUS_AT_LOWER_BOUND;
        } else if (value.has_value() &&
                   std::abs(ub - *value) < kAtBoundTolerance) {
          return BASIS_STATUS_AT_UPPER_BOUND;
        }
        // We cannot infer if we are at upper or at lower. Mathopt does not
        // an encoding for nonbasic but unknown upper/lower, see b/272767311.
        return std::nullopt;
      } else if (lb_finite) {
        return BASIS_STATUS_AT_LOWER_BOUND;
      } else if (ub_finite) {
        return BASIS_STATUS_AT_LOWER_BOUND;
      } else {
        return BASIS_STATUS_FREE;
      }
    }
  }
  return util::InternalErrorBuilder()
         << "unexpected highs basis: " << static_cast<int>(highs_basis);
}

absl::StatusOr<SolutionStatusProto> ToSolutionStatus(
    const HighsInt highs_solution_status) {
  switch (highs_solution_status) {
    case ::kSolutionStatusInfeasible:
      return SOLUTION_STATUS_INFEASIBLE;
    case ::kSolutionStatusFeasible:
      return SOLUTION_STATUS_FEASIBLE;
    case ::kSolutionStatusNone:
      return SOLUTION_STATUS_UNDETERMINED;
  }
  return util::InternalErrorBuilder()
         << "unimplemented highs SolutionStatus: " << highs_solution_status;
}

}  // namespace



#endif





absl::Status MosekSolver::AddVariables(const VariablesProto & vars) {
  const int num_vars = model.variables().ids_size;
  int firstvar; MSK_getnumvar(task,&firstvar);

  { int i = 0; for (const auto & v : vars.ids()) variable[v] = firstvar+i; ++i; }

  std::vector<double> lbx(num_vars);
  std::vector<double> ubx(num_vars);
  { int i = 0; for (const auto & v : vars.lower_bounds()) lbx[i++] = v; }
  { int i = 0; for (const auto & v : vars.upper_bounds()) ubx[i++] = v; }
  
  RETURN_IF_ERROR(msk.AppendVars(lbx,ubx));

  {
    int i = 0; 
    for (const bool is_integer : vars.integers()) {
      if (is_integer) {
        RETURN_IF_ERROR(msk.PutVarType(ids[i],true));
      }
      ++i;
    }
  }
  { 
    int i = 0; for (const auto & name : model.linear_variables().names()) {
      msk.PutVarName(firstvar+i,name.c_str());
      ++i;
    }
  }
} // MosekSolver::AddVariables

absl::Status MosekSolver::ReplaceObjective(const ObjectiveProto & obj) {
  msk.PutObjName(obj.name());
  msk.PutObjectiveSense(model.maximize());
  auto objcof = obj.linear_coefficients();
  msk.PutCFix(obj.offset());
  std::vector<double> c(num_vars);
  auto n = objcof.ids_size;
  for (int64_t i = 0; i < n; ++i) {
    c[objcof.ids(i)] = objcof.values(i);
  }
  msk.PutC(c);
} // MosekSolver::ReplaceObjective

absl::Status MosekSolver::AddConstraints(const ConstraintsProto & cons) {
  int firstcon = ms. MSK_getnumcon(task,&firstcon);
  auto numcon = cons.size_ids;
  {
    int i = 0;
    for (const auto& id : cons.ids()) {
      linconstr_map[id] = i;
      ++i;
    }
  }
  std::vector<double> clb(numcon);
  std::vector<double> cub(numcon);
  { int i = 0; for (const auto & b : cons.lower_bounds()) clb[i++] = b; }
  { int i = 0; for (const auto & b : cons.upper_bounds()) cub[i++] = b; }
  msg.AppendCons(clb,cub);

  { 
    int i = 0; for (const auto & name : cons.names()) {
      msk.PutConName(firstcon+i,name.c_str());
      ++i;
    }
  }

  auto adata = model.linear_constraint_matrix();
  size_t nnz = adata.size_row_ids;
  std::vector<Mosek::VariableIndex> subj; subj.reserve(nnz);
  std::vector<Mosek::ConstraintIndex> subi; subi.reserve(nnz);
  std::vector<double> valij; valij.reserve(nnz);

  for (const auto id : adata.row_ids())
    subj.push_back(variable_map[id]);
  for (const auto id : adata.column_ids())
    subi.push_back(linconstr_map[id]);
  for (const auto c : adata.coefficients())
    valij.push_back(c);
  RETURN_IF_ERROR(msk.PutAIJList(asubi,asubj,valij));

  return absl::OkStatus();
} // MosekSolver::AddConstraints

absl::Status MosekSolver::AddIndicatorConstraints(
    const ::google::protobuf::Map<int64_t, IndicatorConstraintProto>& cons) {
  int i = 0;
  std::vector<Mosek::VariableIndex> subj;
  std::vector<double> cof;
  for (const auto & [id, con] : cons) {
    indconstr_map[id] = i++;
    Mosek::VariableIndex indvar = indconstr_map[con.indicator_id()];

    subj.clear(); subj.reserve(con.expressions().size_ids);
    cof.clear(); cof.reserve(con.expressions().size_ids);

    for (auto id : con.expression().ids()) { subj.push_back(variable_map[id]); }
    for (auto c : con.expression().values() { cof.push_back(c); }

    auto djci = msk.AppendIndicatorConstraint(con.activate_on_zero(),indvar,subj,cof,con.lower_bound(),con.upper_bound());
    if (!djci.ok()) { return djci.status(); }

    msk.PutDJCName(*djci,con.name());
  }
  return absl::OkStatus();
} // MosekSolver::AddIndicatorConstraints

absl::Status MosekSolver::AddConicConstraints(
  const ::google::protobuf::Map<int64_t, SecondOrderCodeConstraintProto>&
      cons) {

  std::vector<Mosek::VariableIndex> subj;
  std::vector<double>  cof;
  std::vector<int64_t> ptr;
  std::vector<double>  b;

  ptr.reserve(1+cons.size());
  ptr.push_back(0);
  for (const auto & [idx, con] : cons) {
    auto & expr0 = con.upper_bound();
    int64_t totalnnz = expr0.size_ids();
    for (const auto & lexp : con.arguments_to_norm()) {
      totalnnz += lexp.size_ids();
    }

    subj.reserve(totalnnz);
    cof.reserve(totalnnz);
    b.push_back(expr0.offset());

    for (const auto & id : expr0.ids()) { subj.push_back(variable_map[id]); }
    for (auto c : expr0.coefficients()) { cof.push_back(c); }

    for (const auto & expri : con.arguments_to_norm()) {
      ptr.push_back(subj.size());
      for (const auto & id : expri.ids()) { subj.push_back(variable_map[id]); }
      for (auto c : expri.coefficients()) { cof.push_back(c); }
      b.push_back(expri.offset());
    }

    ptr.push_back(subj.size());

    auto acci = msk.AppendConeConstraint(Mosek::ConeType::SecondOrderCone,ptr,subj,cof,b);
    if (!acci.ok()) {
      return acci.status();
    }

    RETURN_IF_ERROR(msk.PutAccName(*acci, con.name()));
  }
  return absl::OkStatus();
}












//absl::StatusOr<FeasibilityStatusProto> MosekSolver::DualFeasibilityStatus(
//    const HighsInfo& highs_info, const bool is_integer,
//    const SolutionClaims solution_claims) {
//  const bool dual_feasible_solution_exists =
//      solution_claims.highs_returned_dual_feasible_solution ||
//      (is_integer && std::isfinite(highs_info.mip_dual_bound));
//  if (dual_feasible_solution_exists &&
//      solution_claims.highs_returned_primal_ray) {
//    return util::InternalErrorBuilder()
//           << "Found dual feasible solution and primal ray";
//  }
//  if (dual_feasible_solution_exists) {
//    return FEASIBILITY_STATUS_FEASIBLE;
//  }
//  if (solution_claims.highs_returned_primal_ray) {
//    return FEASIBILITY_STATUS_INFEASIBLE;
//  }
//  return FEASIBILITY_STATUS_UNDETERMINED;
//}
//
//absl::StatusOr<FeasibilityStatusProto> MosekSolver::PrimalFeasibilityStatus(
//    const SolutionClaims solution_claims) {
//  if (solution_claims.highs_returned_primal_feasible_solution &&
//      solution_claims.highs_returned_dual_ray) {
//    return util::InternalErrorBuilder()
//           << "Found primal feasible solution and dual ray";
//  }
//  if (solution_claims.highs_returned_primal_feasible_solution) {
//    return FEASIBILITY_STATUS_FEASIBLE;
//  }
//  if (solution_claims.highs_returned_dual_ray) {
//    return FEASIBILITY_STATUS_INFEASIBLE;
//  }
//  return FEASIBILITY_STATUS_UNDETERMINED;
//}
//
//absl::StatusOr<TerminationProto> MosekSolver::MakeTermination(
//    const HighsModelStatus highs_model_status, const HighsInfo& highs_info,
//    const bool is_integer, const bool had_node_limit,
//    const bool had_solution_limit, const bool is_maximize,
//    const SolutionClaims solution_claims) {
//  ASSIGN_OR_RETURN(
//      const FeasibilityStatusProto dual_feasibility_status,
//      DualFeasibilityStatus(highs_info, is_integer, solution_claims));
//  ASSIGN_OR_RETURN(const FeasibilityStatusProto primal_feasibility_status,
//                   PrimalFeasibilityStatus(solution_claims));
//
//  const std::optional<double> optional_finite_primal_objective =
//      (primal_feasibility_status == FEASIBILITY_STATUS_FEASIBLE)
//          ? std::make_optional(highs_info.objective_function_value)
//          : std::nullopt;
//  const std::optional<double> optional_dual_objective =
//      (dual_feasibility_status == FEASIBILITY_STATUS_FEASIBLE)
//          ? std::make_optional(DualObjective(highs_info, is_integer))
//          : std::nullopt;
//  switch (highs_model_status) {
//    case HighsModelStatus::kNotset:
//    case HighsModelStatus::kLoadError:
//    case HighsModelStatus::kModelError:
//    case HighsModelStatus::kPresolveError:
//    case HighsModelStatus::kSolveError:
//    case HighsModelStatus::kPostsolveError:
//    case HighsModelStatus::kUnknown:
//    // Note: we actually deal with kModelEmpty separately in Solve(), this
//    // case should not be hit.
//    case HighsModelStatus::kModelEmpty:
//      return util::InternalErrorBuilder()
//             << "HighsModelStatus was "
//             << utilModelStatusToString(highs_model_status);
//    case HighsModelStatus::kOptimal: {
//      return OptimalTerminationProto(highs_info.objective_function_value,
//                                     DualObjective(highs_info, is_integer),
//                                     "HighsModelStatus is kOptimal");
//    }
//    case HighsModelStatus::kInfeasible:
//      // By convention infeasible MIPs are always dual feasible.
//      return InfeasibleTerminationProto(is_maximize,
//                                        /*dual_feasibility_status=*/is_integer
//                                            ? FEASIBILITY_STATUS_FEASIBLE
//                                            : dual_feasibility_status);
//    case HighsModelStatus::kUnboundedOrInfeasible:
//      return InfeasibleOrUnboundedTerminationProto(
//          is_maximize, dual_feasibility_status,
//          "HighsModelStatus is kUnboundedOrInfeasible");
//    case HighsModelStatus::kUnbounded: {
//      // TODO(b/271104776): we should potentially always return
//      // TERMINATION_REASON_UNBOUNDED instead, we need to determine if
//      // HighsModelStatus::kUnbounded implies the problem is known to be primal
//      // feasible (for LP and MIP).
//      if (highs_info.primal_solution_status == ::kSolutionStatusFeasible) {
//        return UnboundedTerminationProto(is_maximize);
//      } else {
//        return InfeasibleOrUnboundedTerminationProto(
//            is_maximize,
//            /*dual_feasibility_status=*/FEASIBILITY_STATUS_INFEASIBLE,
//            "HighsModelStatus is kUnbounded");
//      }
//    }
//    case HighsModelStatus::kObjectiveBound:
//      return LimitTerminationProto(
//          is_maximize, LIMIT_OBJECTIVE, optional_finite_primal_objective,
//          optional_dual_objective, "HighsModelStatus is kObjectiveBound");
//    case HighsModelStatus::kObjectiveTarget:
//      return LimitTerminationProto(
//          is_maximize, LIMIT_OBJECTIVE, optional_finite_primal_objective,
//          optional_dual_objective, "HighsModelStatus is kObjectiveTarget");
//    case HighsModelStatus::kTimeLimit:
//      return LimitTerminationProto(is_maximize, LIMIT_TIME,
//                                   optional_finite_primal_objective,
//                                   optional_dual_objective);
//    case HighsModelStatus::kIterationLimit: {
//      if (is_integer) {
//        if (had_node_limit && had_solution_limit) {
//          return LimitTerminationProto(
//              is_maximize, LIMIT_UNDETERMINED, optional_finite_primal_objective,
//              optional_dual_objective,
//              "Both node limit and solution limit were requested, cannot "
//              "determine reason for termination");
//        } else if (had_node_limit) {
//          return LimitTerminationProto(is_maximize, LIMIT_NODE,
//                                       optional_finite_primal_objective,
//                                       optional_dual_objective);
//        } else if (had_solution_limit) {
//          return LimitTerminationProto(is_maximize, LIMIT_SOLUTION,
//                                       optional_finite_primal_objective,
//                                       optional_dual_objective);
//        }
//      } else {
//        // For LP, only the MathOpt iteration limit can cause highs to return
//        // HighsModelStatus::kIterationLimit.
//        return LimitTerminationProto(is_maximize, LIMIT_ITERATION,
//                                     optional_finite_primal_objective,
//                                     optional_dual_objective);
//      }
//    }
//  }
//  return util::InternalErrorBuilder() << "HighsModelStatus unimplemented: "
//                                      << static_cast<int>(highs_model_status);
//}
//
//SolveResultProto MosekSolver::ResultForHighsModelStatusModelEmpty(
//    const bool is_maximize, const double objective_offset,
//    const absl::flat_hash_map<int64_t, IndexAndBound>& lin_con_data) {
//  SolveResultProto result;
//  bool feasible = true;
//  for (const auto& [unused, lin_con_bounds] : lin_con_data) {
//    if (lin_con_bounds.lb > 0 || lin_con_bounds.ub < 0) {
//      feasible = false;
//      break;
//    }
//  }
//  result.mutable_termination()->set_reason(
//      feasible ? TERMINATION_REASON_OPTIMAL : TERMINATION_REASON_INFEASIBLE);
//  result.mutable_termination()->set_detail("HighsModelStatus was kEmptyModel");
//  if (feasible) {
//    auto solution = result.add_solutions()->mutable_primal_solution();
//    solution->set_objective_value(objective_offset);
//    solution->set_feasibility_status(SOLUTION_STATUS_FEASIBLE);
//    *result.mutable_termination() =
//        OptimalTerminationProto(objective_offset, objective_offset);
//  } else {
//    // If the primal problem has no variables, the dual problem is unconstrained
//    // and thus always feasible.
//    *result.mutable_termination() =
//        InfeasibleTerminationProto(is_maximize, /*dual_feasibility_status=*/
//                                   FEASIBILITY_STATUS_FEASIBLE);
//    // It is probably possible to return a ray here as well.
//  }
//  return result;
//}
//
//InvertedBounds MosekSolver::ListInvertedBounds() {
//  const auto find_crossed =
//      [](const absl::flat_hash_map<int64_t, IndexAndBound>& id_to_bound_data) {
//        std::vector<int64_t> result;
//        for (const auto& [id, bound_data] : id_to_bound_data) {
//          if (bound_data.bounds_cross()) {
//            result.push_back(id);
//          }
//        }
//        absl::c_sort(result);
//        return result;
//      };
//  return {.variables = find_crossed(variable_data_),
//          .linear_constraints = find_crossed(lin_con_data_)};
//}
//
//absl::StatusOr<std::optional<BasisProto>> MosekSolver::ExtractBasis() {
//  const HighsInfo& highs_info = highs_->getInfo();
//  const HighsBasis& highs_basis = highs_->getBasis();
//  const HighsSolution& highs_solution = highs_->getSolution();
//  if (highs_info.basis_validity != ::kBasisValidityValid) {
//    return std::nullopt;
//  }
//  // We need the primal/dual solution to try and infer a more precise status
//  // for varaiables and constraints listed as kNonBasic.
//  if (!highs_solution.value_valid || !highs_solution.dual_valid) {
//    return std::nullopt;
//  }
//  // Make sure the solution is the right size
//  RETURN_IF_ERROR(EnsureOneEntryPerVariable(highs_solution.col_value))
//      << "invalid highs_solution.col_value";
//  RETURN_IF_ERROR(EnsureOneEntryPerVariable(highs_solution.col_dual))
//      << "invalid highs_solution.col_dual";
//  // Make sure the basis is the right size
//  RETURN_IF_ERROR(EnsureOneEntryPerVariable(highs_basis.col_status))
//      << "invalid highs_basis.col_status";
//  RETURN_IF_ERROR(EnsureOneEntryPerLinearConstraint(highs_basis.row_status))
//      << "invalid highs_basis.row_status";
//  BasisProto basis;
//
//  if (highs_->getModelStatus() == HighsModelStatus::kOptimal) {
//    basis.set_basic_dual_feasibility(SOLUTION_STATUS_FEASIBLE);
//  } else if (highs_info.dual_solution_status == kSolutionStatusInfeasible) {
//    basis.set_basic_dual_feasibility(SOLUTION_STATUS_INFEASIBLE);
//  } else {
//    // TODO(b/272767311): we need to do more to fill this in properly.
//    basis.set_basic_dual_feasibility(SOLUTION_STATUS_UNDETERMINED);
//  }
//  for (const int64_t var_id : SortedMapKeys(variable_data_)) {
//    const IndexAndBound& index_and_bounds = variable_data_.at(var_id);
//    const double var_value = highs_solution.col_value[index_and_bounds.index];
//    OR_ASSIGN_OR_RETURN3(
//        const std::optional<BasisStatusProto> status,
//        ToBasisStatus(highs_basis.col_status[variable_data_.at(var_id).index],
//                      index_and_bounds.lb, index_and_bounds.ub, var_value),
//        _ << "invalid highs_basis.col_status for variable with id: " << var_id);
//    if (!status.has_value()) {
//      return std::nullopt;
//    }
//    basis.mutable_variable_status()->add_ids(var_id);
//    basis.mutable_variable_status()->add_values(*status);
//  }
//  for (const int64_t lin_con_id : SortedMapKeys(lin_con_data_)) {
//    const IndexAndBound& index_and_bounds = lin_con_data_.at(lin_con_id);
//    const double dual_value = highs_solution.row_dual[index_and_bounds.index];
//    OR_ASSIGN_OR_RETURN3(
//        const std::optional<BasisStatusProto> status,
//        ToBasisStatus(highs_basis.row_status[index_and_bounds.index],
//                      index_and_bounds.lb, index_and_bounds.ub, dual_value),
//        _ << "invalid highs_basis.row_status for linear constraint with id: "
//          << lin_con_id);
//    if (!status.has_value()) {
//      return std::nullopt;
//    }
//    basis.mutable_constraint_status()->add_ids(lin_con_id);
//    basis.mutable_constraint_status()->add_values(*status);
//  }
//  return basis;
//}
//
//absl::StatusOr<bool> MosekSolver::PrimalRayReturned() const {
//  if (!highs_->hasInvert()) {
//    return false;
//  }
//  bool has_primal_ray = false;
//  // Note getPrimalRay may return without modifying has_primal_ray, in which
//  // case it will remain at its default false value.
//  RETURN_IF_ERROR(ToStatus(highs_->getPrimalRay(has_primal_ray,
//                                                /*primal_ray_value=*/nullptr)));
//  return has_primal_ray;
//}
//
//absl::StatusOr<bool> MosekSolver::DualRayReturned() const {
//  if (!highs_->hasInvert()) {
//    return false;
//  }
//  bool has_dual_ray = false;
//  // Note getPrimalRay may return without modifying has_dual_ray, in which
//  // case it will remain at its default false value.
//  RETURN_IF_ERROR(ToStatus(highs_->getDualRay(has_dual_ray,
//                                              /*dual_ray_value=*/nullptr)));
//  return has_dual_ray;
//}
//
//absl::StatusOr<MosekSolver::SolutionsAndClaims>
//MosekSolver::ExtractSolutionAndRays(
//    const ModelSolveParametersProto& model_params) {
//  const HighsInfo& highs_info = highs_->getInfo();
//  const HighsSolution& highs_solution = highs_->getSolution();
//  SolutionsAndClaims solution_and_claims;
//  if (highs_info.primal_solution_status == ::kSolutionStatusFeasible &&
//      !highs_solution.value_valid) {
//    return absl::InternalError(
//        "highs_info.primal_solution_status==::kSolutionStatusFeasible, but no "
//        "valid primal solution returned");
//  }
//  if (highs_solution.value_valid || highs_solution.dual_valid) {
//    SolutionProto& solution =
//        solution_and_claims.solutions.emplace_back(SolutionProto());
//    if (highs_solution.value_valid) {
//      RETURN_IF_ERROR(EnsureOneEntryPerVariable(highs_solution.col_value))
//          << "invalid highs_solution.col_value";
//      PrimalSolutionProto& primal_solution =
//          *solution.mutable_primal_solution();
//      primal_solution.set_objective_value(highs_info.objective_function_value);
//      OR_ASSIGN_OR_RETURN3(const SolutionStatusProto primal_solution_status,
//                           ToSolutionStatus(highs_info.primal_solution_status),
//                           _ << "invalid highs_info.primal_solution_status");
//      primal_solution.set_feasibility_status(primal_solution_status);
//      solution_and_claims.solution_claims
//          .highs_returned_primal_feasible_solution =
//          primal_solution.feasibility_status() == SOLUTION_STATUS_FEASIBLE;
//      for (const int64_t var_id : SortedMapKeys(variable_data_)) {
//        primal_solution.mutable_variable_values()->add_ids(var_id);
//        primal_solution.mutable_variable_values()->add_values(
//            highs_solution.col_value[variable_data_.at(var_id).index]);
//      }
//    }
//    if (highs_solution.dual_valid) {
//      RETURN_IF_ERROR(EnsureOneEntryPerVariable(highs_solution.col_dual))
//          << "invalid highs_solution.col_dual";
//      RETURN_IF_ERROR(
//          EnsureOneEntryPerLinearConstraint(highs_solution.row_dual))
//          << "invalid highs_solution.row_dual";
//      DualSolutionProto& dual_solution = *solution.mutable_dual_solution();
//      dual_solution.set_objective_value(highs_info.objective_function_value);
//      OR_ASSIGN_OR_RETURN3(const SolutionStatusProto dual_solution_status,
//                           ToSolutionStatus(highs_info.dual_solution_status),
//                           _ << "invalid highs_info.dual_solution_status");
//      dual_solution.set_feasibility_status(dual_solution_status);
//      solution_and_claims.solution_claims
//          .highs_returned_dual_feasible_solution =
//          dual_solution.feasibility_status() == SOLUTION_STATUS_FEASIBLE;
//      for (const int64_t var_id : SortedMapKeys(variable_data_)) {
//        dual_solution.mutable_reduced_costs()->add_ids(var_id);
//        dual_solution.mutable_reduced_costs()->add_values(
//            highs_solution.col_dual[variable_data_.at(var_id).index]);
//      }
//      for (const int64_t lin_con_id : SortedMapKeys(lin_con_data_)) {
//        dual_solution.mutable_dual_values()->add_ids(lin_con_id);
//        dual_solution.mutable_dual_values()->add_values(
//            highs_solution.row_dual[lin_con_data_.at(lin_con_id).index]);
//      }
//    }
//    ASSIGN_OR_RETURN(std::optional<BasisProto> basis_proto,
//                     MosekSolver::ExtractBasis());
//    if (basis_proto.has_value()) {
//      *solution.mutable_basis() = *std::move(basis_proto);
//    }
//    ApplyAllFilters(model_params, solution);
//  }
//
//  ASSIGN_OR_RETURN(
//      solution_and_claims.solution_claims.highs_returned_primal_ray,
//      PrimalRayReturned());
//  ASSIGN_OR_RETURN(solution_and_claims.solution_claims.highs_returned_dual_ray,
//                   DualRayReturned());
//
//  return solution_and_claims;
//}

// Notes on `Update()`.
//
// # Deleting items
// We will not actually delete variables or constraints from the Task since that
// will change indexes of all successive items. Instead we will remove non-zeros
// and bounds from the deleted items. 
absl::StatusOr<bool> MosekModel::Update(const ModelUpdateProto& model_update) {
  for (auto id : model_update.deleted_variable_ids()) msk.ClearVariable(variable_map[id]);
  for (auto id : model_update.deleted_linear_constraint_ids()) msk.ClearConstraint(linconstr_map[id]);

  if (!AddVariables(model_update.new_variables()).ok()) return false;
  if (!Variables(model_update.new_variables())) return false;
  if (!UpdateVariables(model_update.variable_updates())) return false;
  if (!AddConstraints(model_update.new_linear_constraints())) return false;
  if (!UpdateConstraints(model_update.linear_constraint_updates(),
        model_update.linear_constraint_matrix_updates())) return false;
  if (!UpdateObjective(mosek_update.objective_updates())) return false;
  for (const auto & conupd : model_update.second_order_cone_constraint_updates()) {
    if (!UpdateConeConstraint(conupd)) return false;
  }
  for (const auto & conupd : model_update.indicator_constraint_updates()) {
    if (!UpdateIndicatorConstraint(conupd)) return false;
  }
  return true;
}

absl::Status MosekModel::UpdateVariables(const VariableUpdatesProto & varupds) {
  for (int64_t i = 0, n = varupds.lower_bounds().size_ids; i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateVarLowerBound(variable_map[varupds.lower_bounds().ids(i)], varupds.lower_bounds().values(i)));
  }
  for (int64_t i = 0, n = varupds.upper_bounds().size_ids; i < n; ++i) {
    RETURN_OF_ERROR(msk.UpdateVarUpperBound(variable_map[varupds.upper_bounds().ids(i)], varupds.upper_bounds().values(i)));
  }
  for (int64_t i = 0, n = varupds.integers().size_ids; i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateVariableType(variable_map[varupds.upper_bounds().ids(i)], varupds.integer().values(i)));
  }
  return absl::OkStatus();
}
absl::Status MosekModel::UpdateConstraints(const ConstraintsUpdatesProto & conupds, const SparseDoubleMatrixProto & lincofupds) {
  for (int64_t i = 0, n = conupds.lower_bounds().size_ids; i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateConstraintLowerBound(coniable_map[conupds.lower_bounds().ids(i)], conupds.lower_bounds().values(i)));
  }
  for (int64_t i = 0, n = conupds.upper_bounds().size_ids; i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateConstraintUpperBound(coniable_map[conupds.upper_bounds().ids(i)], conupds.upper_bounds().values(i)));
  }

  size_t n = lincofupds.size_ids;
  std::vector<Mosek::ConstaintIndex> subi(n);
  std::vector<Mosek::VariableIndex> subj(n); 
  std::vector<double> valij(lincofupds.coefficients().begin(),lincofupds.coefficients().end());
  { int i = 0; for (auto id : lincofupds.row_ids()) { subi[i] = linconstr_map[id]; ++i; } }
  { int i = 0; for (auto id : lincofupds.column_ids()) { subj[i] = variable_map[id]; ++i; } }

  RETURN_IF_ERROR(msk.UpdateA(subi,subj,valij));
  return absl::OkStatus();
}
absl::Status MosekModel::UpdateObjective(const ObjectiveUpdatesProto & objupds) {
  std::vector<double> cof(objupds.linear_coefficients().begin(),objupds.linear_coefficients().end());
  std::vecor<Mosek::VariableIndex> subj; subj.reserve(cof.size());
  for (auto id : objupds.coefficients().ids()) subj.push_back(variable_map[id]);

  RETURN_IF_ERROR(msk.UpdateObjectiveSense(objupds.direction_update()));
  RETURN_IF_ERROR(msk.UpdateObjective(objupds.offset_update(),subj,cof));

  return absl::OkStatus();
}
absl::Status MosekModel::UpdateConstraint(const SecondOrderConeConstraintUpdatesProto& conupds) {
  for (auto id : conupds.deleted_constraint_ids()) {
    RETURN_IF_ERROR(msk.ClearConeConstraint(coneconstr_map[id]));
  }

  RETURN_IF_ERROR(AddConicConstraints(conupds.new_constraints()));

  return absl::OkStatus();
}
absl::Status MosekModel::UpdateConstraint(const IndicatorConstraintUpdatesProto& conupds) {
  for (auto id : conupds.deleted_constraint_ids()) {
    RETURN_IF_ERROR(msk.ClearDisjunctiveConstraint(indconstr_map[id]));
  }
  
  RETURN_IF_ERROR(AddConicConstraints(conupds.new_constraints()));

  return absl::OkStatus();
}







absl::StatusOr<std::unique_ptr<SolverInterface>> Mosek::New(
    const ModelProto& model, const InitArgs&) {
  RETURN_IF_ERROR(ModelIsSupported(model, kMosekSupportedStructures, "Mosek"));
  
  if (!input_model.auxiliary_objectives().empty())
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support multi-objective models";
  if (!input_model.objective().quadratic_coefficients().row_ids().empty()) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with quadratic objectives";
  }
  if (!input_model.quadratic_constraints().empty()) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with quadratic constraints";
  }
  if (!input_model.sos1_constraints().empty() || 
      !input_model.sos2_constraints().empty() ) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with SOS constraints";
  }

  std::unique_ptr<MosekSolver> mskslv(new MosekSolver());
  mskslv->msk.PutName(model.name());

  RETURN_IF_ERROR(msk->AddVariables(model.variables()));
  RETURN_IF_ERROR(msk->ReplaceObjective(model.objective()));
  RETURN_IF_ERROR(msk->AddConstraints(model.constraints()));
  RETURN_IF_ERROR(msk->AddIndicatorConstraints(model.indicator_constraints()));

  std::unique_ptr<SolverInterface> res(std::move(mskslv));

  return res;
}

absl::StatusOr<SolveResultProto> MosekSolver::Solve(
    const SolveParametersProto& parameters,
    const ModelSolveParametersProto& model_parameters,
    MessageCallback message_cb, 
    const CallbackRegistrationProto&, 
    Callback cb,
    const SolveInterrupter* const) {
   
  // Solve parameters that we support:
  // - google.protobuf.Duration time_limit
  // - optional int64 iteration_limit
  // - optional int64 node_limit
  // - optional double cutoff_limit
  // - bool enable_output
  // - optional int32 threads
  // - optional double absolute_gap_tolerance
  // - optional double relative_gap_tolerance
  // - LPAlgorithmProto lp_algorithm
  // Solve parameters that we may support:
  // - optional double best_bound_limit
  // - optional double objective_limit
  // Solve parameters that we do not support:
  // - optional int32 solution_pool_size
  // - optional int32 solution_limit
  // - optional int32 random_seed
  // - EmphasisProto presolve
  // - EmphasisProto cuts
  // - EmphasisProto heuristics
  // - EmphasisProto scaling

  
  double dpar_optimizer_max_time = msk.GetParam(MSK_DPAR_OPTIMIZER_MAX_TIME);
  if (parameters.has_time_limit()) {
    OR_ASSIGN_OR_RETURN3(
        const absl::Duration time_limit,
        util_time::DecodeGoogleApiProto(parameters.time_limit()),
        _ << "invalid time_limit value for HiGHS.");
    msk.PutDoubleParam(MSK_DPAR_OPTIMIZER_MAX_TIME, absl::ToDoubleSeconds(time_limit));
  }

  int ipar_intpnt_max_iterations = msk.GetParam(MSK_IPAR_INTPNT_MAX_ITERATIONS);
  int ipar_sim_max_iterations = msk.GetParam(MSK_IPAR_SIM_MAX_ITERATIONS);
  if (parameters.has_iteration_limit()) {
    ASSIGN_OR_RETURN(
        const int iter_limit,
        SafeIntCast(parameters.iteration_limit(), "iteration_limit"));

    msk.PutIntParam(MSK_IPAR_INTPNT_MAX_ITERATIONS, iter_limit);
    msk.PutIntParam(MSK_IPAR_SIM_MAX_ITERATIONS, iter_limit);
  }
  
  // Not supported in MOSEK 10.2
  //int ipar_mio_
  //if (parameters.has_node_limit()) {
  //  ASSIGN_OR_RETURN(
  //      const int node_limit,
  //      SafeIntCast(parameters.node_limit(), "node_limit"));
  //  msk.PutIntParam(MSK_IPAR_MIO__MAX_NODES, node_limit);
  //}
 
  // Not supported by MOSEK?
  //if (parameters.has_cutoff_limit()) {
  //}
  double dpar_upper_obj_cut = msk.GetParam(MSK_DPAR_UPPER_OBJ_CUT);
  double dpar_lower_obj_cut = msk.GetParam(MSK_DPAR_LOWER_OBJ_CUT);
  if (parameters.has_objective_limit()) {
    if (msk.IsMaximize()) 
      msk.PutDoubleParam(MSK_DPAR_UPPER_OBJ_CUT, parameters.cutoff_limit());
    else
      msk.PutDoubleParam(MSK_DPAR_LOWER_OBJ_CUT, parameters.cutoff_limit());
  }

  int ipar_num_threads = msk.GetParam(MSK_IPAR_NUM_THREADS);
  if (parameters.has_threads()) {
    msk.PutIntParam(MSK_IPAR_NUM_THREADS, parameters.threads());
  }

  double dpar_mio_tol_abs_gap = msk.GetParam(MSK_DPAR_MIO_TOL_ABS_GAP);
  if (parameters.has_absolute_gap_tolerance()) {
    msk.PutDoubleParameter(MSK_DPAR_MIO_TOL_ABS_GAP, parameters.absolute_gap_tolerance());
  }
  
  double dpar_mio_tol_rel_gap = msk.GetParam(MSK_DPAR_MIO_TOL_REL_GAP);
  double dpar_intpnt_tol_rel_gap = msk.GetDoubleParam(MSK_DPAR_INTPNT_TOL_REL_GAP);
  double dpar_intpnt_co_tol_rel_gap = msk.GetDoubleParam(MSK_DPAR_INTPNT_CO_TOL_REL_GAP);
  if (parameters.has_relative_gap_tolerance()) {
    msk.PutDoubleParameter(MSK_DPAR_INTPNT_TOL_REL_GAP, parameters.absolute_gap_tolerance());
    msk.PutDoubleParameter(MSK_DPAR_INTPNT_CO_TOL_REL_GAP, parameters.absolute_gap_tolerance());
    msk.PutDoubleParameter(MSK_DPAR_MIO_TOL_REL_GAP, parameters.absolute_gap_tolerance());
  }

  int ipar_optimizer = msk.GetParam(MSK_IPAR_OPTIMIZER);
  switch (parameters.lp_algorithm()) {
    case LP_ALGORITHM_BARRIER:
      msk.PutIntParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT);
      break;
    case LP_ALGORITHM_DUAL_SIMPLEX:
      msk.PutIntParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_DUAL_SIMPLEX);
      break;
    case LP_ALGORITHM_PRIMAL_SIMPLEX:
      msk.PutIntParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_PRIMAL_SIMPLEX);
      break;
    default:
      // use default auto select, usually intpnt
      msk.PutIntParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_FREE);
      break;
  }

  // TODO: parameter enable_output
  // TODO optimize

  // Reset Parameters

  msk.PutParam(MSK_DPAR_OPTIMIZER_MAX_TIME,dpar_optimizer_max_time);
  msk.PutParam(MSK_IPAR_INTPNT_MAX_ITERATIONS,ipar_intpnt_max_iterations);
  msk.PutParam(MSK_IPAR_SIM_MAX_ITERATIONS,ipar_sim_max_iterations);
  msk.PutParam(MSK_DPAR_UPPER_OBJ_CUT,dpar_upper_obj_cut);
  msk.PutParam(MSK_DPAR_LOWER_OBJ_CUT,dpar_lower_obj_cut);
  msk.PutParam(MSK_IPAR_NUM_THREADS,ipar_num_threads);
  msk.PutParam(MSK_DPAR_MIO_TOL_ABS_GAP,dpar_mio_tol_abs_gap);
  msk.PutParam(MSK_DPAR_MIO_TOL_REL_GAP,dpar_mio_tol_rel_gap);
  msk.PutParam(MSK_DPAR_INTPNT_TOL_REL_GAP,dpar_intpnt_tol_rel_gap);
  msk.PutParam(MSK_DPAR_INTPNT_CO_TOL_REL_GAP,dpar_intpnt_co_tol_rel_gap);





#if 0
  const absl::Time start = absl::Now();
  auto set_solve_time = [&start](SolveResultProto& result) -> absl::Status {
    const absl::Duration solve_time = absl::Now() - start;
    OR_ASSIGN_OR_RETURN3(*result.mutable_solve_stats()->mutable_solve_time(),
                         util_time::EncodeGoogleApiProto(solve_time),
                         _ << "error encoding solve_stats.solve_time");
    return absl::OkStatus();
  };

  RETURN_IF_ERROR(ListInvertedBounds().ToStatus());
  // TODO(b/271595607): delete this code once we upgrade HiGHS, if HiGHS does
  // return a proper infeasibility status for models with empty integer bounds.
  const bool is_maximize = highs_->getModel().lp_.sense_ == ObjSense::kMaximize;
  for (const auto& [var_id, bounds] : variable_data_) {
    if (bounds.rounded_bounds_cross()) {
      SolveResultProto result =
          ResultForIntegerInfeasible(is_maximize, var_id, bounds.lb, bounds.ub);
      RETURN_IF_ERROR(set_solve_time(result));
      return result;
    }
  }

  BufferedMessageCallback buffered_message_callback(std::move(message_cb));
  if (buffered_message_callback.has_user_message_callback()) {
    RETURN_IF_ERROR(ToStatus(
        highs_->setLogCallback(&HighsLogCallback, &buffered_message_callback)))
        << "failed to register logging callback";
  }
  auto message_cb_cleanup =
      absl::MakeCleanup([this, &buffered_message_callback]() {
        if (buffered_message_callback.has_user_message_callback()) {
          // As of March 6th, 2023, this code never returns an error (see the
          // HiGHS source). If we really want to be able to recover from errors,
          // more care is needed, as we need to prevent HiGHS from invoking the
          // user callback after this function, since it will not be alive (e.g.
          // wrap the user callback in a new callback that is guarded by an
          // atomic bool that we disable here). Further, to propagate this
          // error, we need a class instead of absl::Cleanup.
          CHECK_OK(ToStatus(highs_->setLogCallback(nullptr, nullptr)));
          buffered_message_callback.Flush();
        }
      });

  bool is_integer = false;
  // NOTE: lp_.integrality_ may be empty if the problem is an LP.
  for (const HighsVarType var_type : highs_->getModel().lp_.integrality_) {
    if (var_type == HighsVarType::kInteger) {
      is_integer = true;
      break;
    }
  }
  auto it = parameters.highs().bool_options().find("solve_relaxation");
  if (it != parameters.highs().bool_options().end() && it->second) {
    is_integer = false;
  }
  ASSIGN_OR_RETURN(
      const std::unique_ptr<HighsOptions> options,
      MakeOptions(parameters,
                  buffered_message_callback.has_user_message_callback(),
                  is_integer));
  RETURN_IF_ERROR(ToStatus(highs_->passOptions(*options)));
  RETURN_IF_ERROR(ToStatus(highs_->run()));
  std::move(message_cb_cleanup).Invoke();
  // When the model is empty, highs_->getInfo() is invalid, so we bail out.
  if (highs_->getModelStatus() == HighsModelStatus::kModelEmpty) {
    SolveResultProto result = ResultForHighsModelStatusModelEmpty(
        is_maximize, highs_->getModel().lp_.offset_, lin_con_data_);
    RETURN_IF_ERROR(set_solve_time(result));
    return result;
  }
  const HighsInfo& info = highs_->getInfo();
  if (!info.valid) {
    return absl::InternalError("HighsInfo not valid");
  }

  SolveResultProto result;
  ASSIGN_OR_RETURN(SolutionsAndClaims solutions_and_claims,
                   ExtractSolutionAndRays(model_parameters));
  for (SolutionProto& solution : solutions_and_claims.solutions) {
    *result.add_solutions() = std::move(solution);
  }
  ASSIGN_OR_RETURN(*result.mutable_termination(),
                   MakeTermination(highs_->getModelStatus(), info, is_integer,
                                   parameters.has_node_limit(),
                                   parameters.has_solution_limit(), is_maximize,
                                   solutions_and_claims.solution_claims));

  ASSIGN_OR_RETURN(*result.mutable_solve_stats(), ToSolveStats(info));

  RETURN_IF_ERROR(set_solve_time(result));
  return result;
#endif
}

absl::StatusOr<ComputeInfeasibleSubsystemResultProto>
MosekSolver::ComputeInfeasibleSubsystem(const SolveParametersProto&,
                                        MessageCallback,
                                        const SolveInterrupter*) {
  return absl::UnimplementedError(
      "HiGHS does not provide a method to compute an infeasible subsystem");
}

MATH_OPT_REGISTER_SOLVER(SOLVER_TYPE_MOSEK, MosekSolver::New);

}  // namespace operations_research::math_opt

