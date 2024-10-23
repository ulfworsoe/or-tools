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


#include "mosek_solver.h"
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

//#include "absl/algorithm/container.h"
//#include "absl/cleanup/cleanup.h"
//#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
//#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
//#include "absl/time/clock.h"
//#include "absl/time/time.h"
#include "ortools/base/protoutil.h"
#include "ortools/base/status_builder.h"
#include "ortools/base/status_macros.h"
#include "ortools/math_opt/core/empty_bounds.h"
#include "ortools/math_opt/core/inverted_bounds.h"
#include "ortools/math_opt/core/math_opt_proto_utils.h"
#include "ortools/math_opt/core/solver_interface.h"
#include "ortools/math_opt/core/sorted.h"
#include "ortools/math_opt/core/sparse_vector_view.h"
#include "ortools/math_opt/infeasible_subsystem.pb.h"
#include "ortools/math_opt/parameters.pb.h"
#include "ortools/math_opt/result.pb.h"
#include "ortools/math_opt/solution.pb.h"
#include "ortools/math_opt/solvers/mosek.pb.h"
#include "ortools/math_opt/solvers/message_callback_data.h"
#include "ortools/util/solve_interrupter.h"
#include "ortools/util/status_macros.h"

namespace operations_research::math_opt {
namespace {

constexpr SupportedProblemStructures kMosekSupportedStructures = {
    .integer_variables     = SupportType::kSupported,
    .second_order_cone_constraints = SupportType::kSupported,
    .indicator_constraints = SupportType::kSupported,
};


}  // namespace




absl::Status MosekSolver::AddVariables(const VariablesProto & vars) {
  std::cout << "MosekSolver::AddVariables()" << std::endl;
  int num_vars = vars.ids_size();
  int firstvar = msk.NumVar();

  { int i = 0; for (const auto & v : vars.ids()) variable_map[v] = firstvar+i; ++i; }

  std::vector<double> lbx(num_vars);
  std::vector<double> ubx(num_vars);
  { int i = 0; for (const auto & v : vars.lower_bounds()) lbx[i++] = v; }
  { int i = 0; for (const auto & v : vars.upper_bounds()) ubx[i++] = v; }
 
  { 
    auto r = msk.AppendVars(lbx,ubx);
    if (!r.ok())
      return r.status();
  }

  {
    int i = 0; 
    for (const bool is_integer : vars.integers()) {
      if (is_integer) {
        RETURN_IF_ERROR(msk.PutVarType(variable_map[i],true));
      }
      ++i;
    }
  }
  { 
    int i = 0; for (const auto & name : vars.names()) {
      msk.PutVarName(firstvar+i,name.c_str());
      ++i;
    }
  }
  return absl::OkStatus();
} // MosekSolver::AddVariables

absl::Status MosekSolver::ReplaceObjective(const ObjectiveProto & obj) {
  std::cout << "MosekSolver::ReplaceObjective()" << std::endl;
  msk.PutObjName(obj.name());
  auto objcof = obj.linear_coefficients();
  msk.PutCFix(obj.offset());
  auto num_vars = msk.NumVar();
  std::vector<double> c(num_vars);
  auto n = objcof.ids_size();
  for (int64_t i = 0; i < n; ++i) {
    c[objcof.ids(i)] = objcof.values(i);
  }
  RETURN_IF_ERROR(msk.PutC(c));
  return absl::OkStatus();
} // MosekSolver::ReplaceObjective

absl::Status MosekSolver::AddConstraints(const LinearConstraintsProto& cons,
                                         const SparseDoubleMatrixProto& adata) {
  std::cout << "MosekSolver::AddConstraints()" << std::endl;
  int firstcon = msk.NumCon();
  auto numcon = cons.ids_size();
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
  {
    auto r = msk.AppendCons(clb,cub);
    if (!r.ok()) return r.status();
  }

  { 
    int i = 0; for (const auto & name : cons.names()) {
      msk.PutConName(firstcon+i,name.c_str());
      ++i;
    }
  }

  size_t nnz = adata.row_ids_size();
  std::vector<Mosek::VariableIndex> subj; subj.reserve(nnz);
  std::vector<Mosek::ConstraintIndex> subi; subi.reserve(nnz);
  std::vector<double> valij; valij.reserve(nnz);

  for (const auto id : adata.row_ids())
    subj.push_back(variable_map[id]);
  for (const auto id : adata.column_ids())
    subi.push_back(linconstr_map[id]);
  for (const auto c : adata.coefficients()) 
    valij.push_back(c);
  RETURN_IF_ERROR(msk.PutAIJList(subi,subj,valij));

  return absl::OkStatus();
}
absl::Status MosekSolver::AddConstraints(const LinearConstraintsProto & cons) {
  std::cout << "MosekSolver::AddConstraints()" << std::endl;
  int firstcon = msk.NumCon();
  auto numcon = cons.ids_size();
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
  {
    auto r = msk.AppendCons(clb,cub);
    if (!r.ok()) return r.status();
  }

  { 
    int i = 0; for (const auto & name : cons.names()) {
      msk.PutConName(firstcon+i,name.c_str());
      ++i;
    }
  }

  return absl::OkStatus();
} // MosekSolver::AddConstraints

absl::Status MosekSolver::AddIndicatorConstraints(
    const ::google::protobuf::Map<int64_t, IndicatorConstraintProto>& cons) {
  std::cout << "MosekSolver::AddIndicatorConstraints()" << std::endl;
  int i = 0;
  std::vector<Mosek::VariableIndex> subj;
  std::vector<double> cof;
  for (const auto & [id, con] : cons) {
    indconstr_map[id] = i++;
    Mosek::VariableIndex indvar = indconstr_map[con.indicator_id()];

    subj.clear(); subj.reserve(con.expression().ids_size());
    cof.clear(); cof.reserve(con.expression().ids_size());

    for (auto id : con.expression().ids()) { subj.push_back(variable_map[id]); }
    for (auto c : con.expression().values()) { cof.push_back(c); }

    auto djci = msk.AppendIndicatorConstraint(con.activate_on_zero(),indvar,subj,cof,con.lower_bound(),con.upper_bound());
    if (!djci.ok()) { return djci.status(); }

    RETURN_IF_ERROR(msk.PutDJCName(*djci,con.name()));
  }
  return absl::OkStatus();
} // MosekSolver::AddIndicatorConstraints

absl::Status MosekSolver::AddConicConstraints(
  const ::google::protobuf::Map<int64_t, SecondOrderConeConstraintProto>&
      cons) {
  std::cout << "MosekSolver::AddConicConstraints()" << std::endl;

  std::vector<Mosek::VariableIndex> subj;
  std::vector<double>  cof;
  std::vector<int32_t> sizes;
  std::vector<double>  b;

  sizes.reserve(cons.size());
  for (const auto & [idx, con] : cons) {
    auto & expr0 = con.upper_bound();
    int64_t totalnnz = expr0.ids_size();
    for (const auto & lexp : con.arguments_to_norm()) {
      totalnnz += lexp.ids_size();
    }

    subj.reserve(totalnnz);
    cof.reserve(totalnnz);
    b.push_back(expr0.offset());

    for (const auto & id : expr0.ids()) { subj.push_back(variable_map[id]); }
    for (auto c : expr0.coefficients()) { cof.push_back(c); }

    for (const auto & expri : con.arguments_to_norm()) {
      sizes.push_back(expri.ids_size());
      for (const auto & id : expri.ids()) { subj.push_back(variable_map[id]); }
      for (auto c : expri.coefficients()) { cof.push_back(c); }
      b.push_back(expri.offset());
    }

    auto acci = msk.AppendConeConstraint(Mosek::ConeType::SecondOrderCone,sizes,subj,cof,b);
    if (!acci.ok()) {
      return acci.status();
    }

    RETURN_IF_ERROR(msk.PutACCName(*acci, con.name()));
  }
  return absl::OkStatus();
}



absl::StatusOr<bool> MosekSolver::Update(const ModelUpdateProto& model_update) {
  std::cout << "MosekSolver::Update()" << std::endl;
  for (auto id : model_update.deleted_variable_ids()) {
    variable_map.erase(id);
    RETURN_IF_ERROR(msk.ClearVariable(variable_map[id]));
  }
  for (auto id : model_update.deleted_linear_constraint_ids()) {
    linconstr_map.erase(id);
    RETURN_IF_ERROR(msk.ClearConstraint(linconstr_map[id]));
  }
  for (auto id : model_update.second_order_cone_constraint_updates().deleted_constraint_ids()) {
    coneconstr_map.erase(id);
    RETURN_IF_ERROR(msk.ClearConeConstraint(coneconstr_map[id]));
  }
  for (auto id : model_update.indicator_constraint_updates().deleted_constraint_ids()) {
    indconstr_map.erase(id);
    RETURN_IF_ERROR(msk.ClearDisjunctiveConstraint(indconstr_map[id]));
  }

  RETURN_IF_ERROR(AddVariables(model_update.new_variables()));
  RETURN_IF_ERROR(UpdateVariables(model_update.variable_updates()));
  RETURN_IF_ERROR(AddConstraints(model_update.new_linear_constraints()));
  RETURN_IF_ERROR(UpdateConstraints(model_update.linear_constraint_updates(),
        model_update.linear_constraint_matrix_updates()));

  RETURN_IF_ERROR(UpdateObjective(model_update.objective_updates()));
  RETURN_IF_ERROR(AddConicConstraints(model_update.second_order_cone_constraint_updates().new_constraints()));
  RETURN_IF_ERROR(AddIndicatorConstraints(model_update.indicator_constraint_updates().new_constraints()));
  //  RETURN_IF_ERROR(UpdateIndicatorConstraint(conupd));
  return true;
}

absl::Status MosekSolver::UpdateVariables(const VariableUpdatesProto & varupds) {
  std::cout << "MosekSolver::UpdateVariables()" << std::endl;
  for (int64_t i = 0, n = varupds.lower_bounds().ids_size(); i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateVariableLowerBound(variable_map[varupds.lower_bounds().ids(i)], varupds.lower_bounds().values(i)));
  }
  for (int64_t i = 0, n = varupds.upper_bounds().ids_size(); i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateVariableUpperBound(variable_map[varupds.upper_bounds().ids(i)], varupds.upper_bounds().values(i)));
  }
  for (int64_t i = 0, n = varupds.integers().ids_size(); i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateVariableType(variable_map[varupds.upper_bounds().ids(i)], varupds.integers().values(i)));
  }
  return absl::OkStatus();
}
absl::Status MosekSolver::UpdateConstraints(const LinearConstraintUpdatesProto & conupds, const SparseDoubleMatrixProto & lincofupds) {
  std::cout << "MosekSolver::UpdateConstraints()" << std::endl;
  for (int64_t i = 0, n = conupds.lower_bounds().ids_size(); i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateConstraintLowerBound(linconstr_map[conupds.lower_bounds().ids(i)], conupds.lower_bounds().values(i)));
  }
  for (int64_t i = 0, n = conupds.upper_bounds().ids_size(); i < n; ++i) {
    RETURN_IF_ERROR(msk.UpdateConstraintUpperBound(linconstr_map[conupds.upper_bounds().ids(i)], conupds.upper_bounds().values(i)));
  }

  size_t n = lincofupds.row_ids_size();
  std::vector<int> subi(n);
  std::vector<int> subj(n); 
  std::vector<double> valij(lincofupds.coefficients().begin(),lincofupds.coefficients().end());
  { int i = 0; for (auto id : lincofupds.row_ids()) { subi[i] = linconstr_map[id]; ++i; } }
  { int i = 0; for (auto id : lincofupds.column_ids()) { subj[i] = variable_map[id]; ++i; } }

  RETURN_IF_ERROR(msk.UpdateA(subi,subj,valij));
  return absl::OkStatus();
}
absl::Status MosekSolver::UpdateObjective(const ObjectiveUpdatesProto & objupds) {
  const auto& vals = objupds.linear_coefficients();
  std::vector<double> cof(vals.values().begin(),vals.values().end());
  std::vector<Mosek::VariableIndex> subj; subj.reserve(cof.size());
  for (auto id : objupds.linear_coefficients().ids()) subj.push_back(variable_map[id]);

  RETURN_IF_ERROR(msk.UpdateObjectiveSense(objupds.direction_update()));
  RETURN_IF_ERROR(msk.UpdateObjective(objupds.offset_update(),subj,cof));

  return absl::OkStatus();
}
absl::Status MosekSolver::UpdateConstraint(const SecondOrderConeConstraintUpdatesProto& conupds) {
  for (auto id : conupds.deleted_constraint_ids()) {
    RETURN_IF_ERROR(msk.ClearConeConstraint(coneconstr_map[id]));
  }

  RETURN_IF_ERROR(AddConicConstraints(conupds.new_constraints()));

  return absl::OkStatus();
}

absl::Status MosekSolver::UpdateConstraint(const IndicatorConstraintUpdatesProto& conupds) {
  for (auto id : conupds.deleted_constraint_ids()) {
    RETURN_IF_ERROR(msk.ClearDisjunctiveConstraint(indconstr_map[id]));
  }
  
  RETURN_IF_ERROR(AddIndicatorConstraints(conupds.new_constraints()));

  return absl::OkStatus();
}







absl::StatusOr<std::unique_ptr<SolverInterface>> MosekSolver::New(
    const ModelProto& model, const InitArgs&) {
  std::cout << "MosekSolver::New()" << std::endl;
  RETURN_IF_ERROR(ModelIsSupported(model, kMosekSupportedStructures, "Mosek"));
  
  if (!model.auxiliary_objectives().empty())
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support multi-objective models";
  if (!model.objective().quadratic_coefficients().row_ids().empty()) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with quadratic objectives";
  }
  if (!model.quadratic_constraints().empty()) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with quadratic constraints";
  }
  if (!model.sos1_constraints().empty() || 
      !model.sos2_constraints().empty() ) {
    return util::InvalidArgumentErrorBuilder()
           << "Mosek does not support models with SOS constraints";
  }
  
  std::unique_ptr<Mosek> msk(Mosek::Create());
  std::unique_ptr<MosekSolver> mskslv(new MosekSolver(std::move(*msk)));
  mskslv->msk.PutName(model.name());
  //mskslv->msk.UpdateObjectiveSense(model.objective().maximize());

  RETURN_IF_ERROR(mskslv->AddVariables(model.variables()));
  RETURN_IF_ERROR(mskslv->ReplaceObjective(model.objective()));
  RETURN_IF_ERROR(mskslv->AddConstraints(model.linear_constraints(),model.linear_constraint_matrix()));
  RETURN_IF_ERROR(mskslv->AddIndicatorConstraints(model.indicator_constraints()));

  std::unique_ptr<SolverInterface> res(std::move(mskslv));

  return res;
}

MosekSolver::MosekSolver(Mosek && msk) : msk(std::move(msk)) {}
  



absl::StatusOr<PrimalSolutionProto> MosekSolver::PrimalSolution(MSKsoltypee whichsol) {
  auto solsta = msk.GetSolSta(whichsol);
  PrimalSolutionProto sol;
  switch (solsta) {
    case MSK_SOL_STA_OPTIMAL:         
    case MSK_SOL_STA_INTEGER_OPTIMAL: 
    case MSK_SOL_STA_PRIM_AND_DUAL_FEAS:
    case MSK_SOL_STA_PRIM_FEAS:
      sol.set_feasibility_status(SolutionStatusProto::SOLUTION_STATUS_FEASIBLE); 
      {
        sol.set_objective_value(msk.GetPrimalObj(whichsol));
        std::vector<double> xx; msk.GetXX(whichsol,xx);
        SparseDoubleVectorProto vals;
        for (auto &[k,v] : variable_map) 
        {
          vals.add_ids(k);
          vals.add_values(xx[v]);
        }

        *sol.mutable_variable_values() = std::move(vals);
      }
      break;
    default:
      return absl::NotFoundError("Primal solution not available");
  }
  return std::move(sol);
}
absl::StatusOr<DualSolutionProto> MosekSolver::DualSolution(MSKsoltypee whichsol) {
  auto solsta = msk.GetSolSta(whichsol);
  DualSolutionProto sol;
  switch (solsta) {
    case MSK_SOL_STA_OPTIMAL: 
    case MSK_SOL_STA_PRIM_AND_DUAL_FEAS:
    case MSK_SOL_STA_DUAL_FEAS:
      sol.set_objective_value(msk.GetPrimalObj(whichsol));
      sol.set_feasibility_status(SolutionStatusProto::SOLUTION_STATUS_FEASIBLE); 
      {
        std::vector<double> slx; msk.GetSLX(whichsol,slx);
        std::vector<double> sux; msk.GetSUX(whichsol,sux);
        SparseDoubleVectorProto vals;
        for (auto &[k,v] : variable_map) 
        {
          vals.add_ids(k);
          vals.add_values(slx[v]-sux[v]);
        }

        *sol.mutable_dual_values() = std::move(vals);
      }
      {
        std::vector<double> y; msk.GetY(whichsol,y);
        SparseDoubleVectorProto vals;
        for (auto &[k,v] : linconstr_map) 
        {
          vals.add_ids(k);
          vals.add_values(y[v]);
        }

        *sol.mutable_reduced_costs() = std::move(vals);
      }
      break;
    default:
      return absl::NotFoundError("Primal solution not available");
      break;
  }
  return std::move(sol);
}
absl::StatusOr<SolutionProto>  MosekSolver::Solution(MSKsoltypee whichsol) {
  std::cout << "MosekSolver::Solution()" << std::endl;
  SolutionProto sol;
  {
    auto r = PrimalSolution(whichsol);
    if (r.ok())
      *sol.mutable_primal_solution() = std::move(*r);
  }
  {
    auto r = DualSolution(whichsol);
    if (r.ok())
      *sol.mutable_dual_solution() = std::move(*r);
  }
  if (whichsol == MSK_SOL_BAS) {
    BasisProto bas;
    SparseBasisStatusVector csta;
    SparseBasisStatusVector xsta;
    std::vector<MSKstakeye> sk; msk.GetSKX(whichsol,sk);
    for (auto & [k,v] : variable_map) {
      xsta.add_ids(k);
      switch (sk[v]) {
        case MSK_SK_LOW: xsta.add_values(BasisStatusProto::BASIS_STATUS_AT_LOWER_BOUND); break;
        case MSK_SK_UPR: xsta.add_values(BasisStatusProto::BASIS_STATUS_AT_UPPER_BOUND); break;
        case MSK_SK_FIX: xsta.add_values(BasisStatusProto::BASIS_STATUS_FIXED_VALUE); break;
        case MSK_SK_BAS: xsta.add_values(BasisStatusProto::BASIS_STATUS_BASIC); break;
        case MSK_SK_INF: 
        case MSK_SK_SUPBAS:
        case MSK_SK_UNK: xsta.add_values(BasisStatusProto::BASIS_STATUS_UNSPECIFIED); break;
      }
    }
    sk.clear(); msk.GetSKC(whichsol,sk);
    for (auto & [k,v] : linconstr_map) {
      csta.add_ids(k);
      switch (sk[v]) {
        case MSK_SK_LOW: csta.add_values(BasisStatusProto::BASIS_STATUS_AT_LOWER_BOUND); break;
        case MSK_SK_UPR: csta.add_values(BasisStatusProto::BASIS_STATUS_AT_UPPER_BOUND); break;
        case MSK_SK_FIX: csta.add_values(BasisStatusProto::BASIS_STATUS_FIXED_VALUE); break;
        case MSK_SK_BAS: csta.add_values(BasisStatusProto::BASIS_STATUS_BASIC); break;
        case MSK_SK_INF: 
        case MSK_SK_SUPBAS:
        case MSK_SK_UNK: csta.add_values(BasisStatusProto::BASIS_STATUS_UNSPECIFIED); break;
      }
    }
    *bas.mutable_variable_status() = std::move(xsta);
    *bas.mutable_constraint_status() = std::move(csta);

    *sol.mutable_basis() = std::move(bas);
  }
  return std::move(sol);
}

absl::StatusOr<PrimalRayProto> MosekSolver::PrimalRay(MSKsoltypee whichsol) {
  auto solsta = msk.GetSolSta(whichsol);
  if (solsta == MSK_SOL_STA_DUAL_INFEAS_CER)
    return absl::NotFoundError("Certificate not available");

  std::vector<double> xx; msk.GetXX(whichsol,xx);
  PrimalRayProto ray;
  SparseDoubleVectorProto data;
  for (auto &[k,v] : variable_map) {
    data.add_ids(k);
    data.add_values(xx[v]);
  }
  *ray.mutable_variable_values() = data;
  return ray;
}

absl::StatusOr<DualRayProto>   MosekSolver::DualRay(MSKsoltypee whichsol) {
  auto solsta = msk.GetSolSta(whichsol);

  if (solsta == MSK_SOL_STA_PRIM_INFEAS_CER)
    return absl::NotFoundError("Certificate not available");

  std::vector<double> slx; msk.GetSLX(whichsol,slx);
  std::vector<double> sux; msk.GetSUX(whichsol,slx);
  std::vector<double> y; msk.GetY(whichsol,y);
  DualRayProto ray;
  SparseDoubleVectorProto xdata;
  SparseDoubleVectorProto cdata;
  for (auto &[k,v] : variable_map) {
    xdata.add_ids(k);
    xdata.add_values(slx[v] - sux[v]);
  }
  for (auto &[k,v] : linconstr_map) {
    cdata.add_ids(k);
    cdata.add_values(y[v]);
  }
  *ray.mutable_dual_values() = xdata;
  *ray.mutable_reduced_costs() = cdata;
  return ray;
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

  std::cout << "MosekSolver::Solve()" << std::endl;
  
  double dpar_optimizer_max_time = msk.GetParam(MSK_DPAR_OPTIMIZER_MAX_TIME);
  int ipar_intpnt_max_iterations = msk.GetParam(MSK_IPAR_INTPNT_MAX_ITERATIONS);
  int ipar_sim_max_iterations = msk.GetParam(MSK_IPAR_SIM_MAX_ITERATIONS);
  double dpar_upper_obj_cut = msk.GetParam(MSK_DPAR_UPPER_OBJ_CUT);
  double dpar_lower_obj_cut = msk.GetParam(MSK_DPAR_LOWER_OBJ_CUT);
  int ipar_num_threads = msk.GetParam(MSK_IPAR_NUM_THREADS);
  double dpar_mio_tol_abs_gap = msk.GetParam(MSK_DPAR_MIO_TOL_ABS_GAP);
  double dpar_mio_tol_rel_gap = msk.GetParam(MSK_DPAR_MIO_TOL_REL_GAP);
  double dpar_intpnt_tol_rel_gap = msk.GetParam(MSK_DPAR_INTPNT_TOL_REL_GAP);
  double dpar_intpnt_co_tol_rel_gap = msk.GetParam(MSK_DPAR_INTPNT_CO_TOL_REL_GAP);
  int ipar_optimizer = msk.GetParam(MSK_IPAR_OPTIMIZER);

  auto _guard_reset_params = absl::MakeCleanup([&](){
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
  });

  if (parameters.has_time_limit()) {
    OR_ASSIGN_OR_RETURN3(
        const absl::Duration time_limit,
        util_time::DecodeGoogleApiProto(parameters.time_limit()),
        _ << "invalid time_limit value for HiGHS.");
    msk.PutParam(MSK_DPAR_OPTIMIZER_MAX_TIME, absl::ToDoubleSeconds(time_limit));
  }

  if (parameters.has_iteration_limit()) {
    const int iter_limit = parameters.iteration_limit();

    msk.PutParam(MSK_IPAR_INTPNT_MAX_ITERATIONS, iter_limit);
    msk.PutParam(MSK_IPAR_SIM_MAX_ITERATIONS, iter_limit);
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
  if (parameters.has_objective_limit()) {
    if (msk.IsMaximize()) 
      msk.PutParam(MSK_DPAR_UPPER_OBJ_CUT, parameters.cutoff_limit());
    else
      msk.PutParam(MSK_DPAR_LOWER_OBJ_CUT, parameters.cutoff_limit());
  }

  if (parameters.has_threads()) {
    msk.PutParam(MSK_IPAR_NUM_THREADS, parameters.threads());
  }

  if (parameters.has_absolute_gap_tolerance()) {
    msk.PutParam(MSK_DPAR_MIO_TOL_ABS_GAP, parameters.absolute_gap_tolerance());
  }
  
  if (parameters.has_relative_gap_tolerance()) {
    msk.PutParam(MSK_DPAR_INTPNT_TOL_REL_GAP, parameters.absolute_gap_tolerance());
    msk.PutParam(MSK_DPAR_INTPNT_CO_TOL_REL_GAP, parameters.absolute_gap_tolerance());
    msk.PutParam(MSK_DPAR_MIO_TOL_REL_GAP, parameters.absolute_gap_tolerance());
  }

  switch (parameters.lp_algorithm()) {
    case LP_ALGORITHM_BARRIER:
      msk.PutParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_INTPNT);
      break;
    case LP_ALGORITHM_DUAL_SIMPLEX:
      msk.PutParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_DUAL_SIMPLEX);
      break;
    case LP_ALGORITHM_PRIMAL_SIMPLEX:
      msk.PutParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_PRIMAL_SIMPLEX);
      break;
    default:
      // use default auto select, usually intpnt
      msk.PutParam(MSK_IPAR_OPTIMIZER, MSK_OPTIMIZER_FREE);
      break;
  }

  // TODO: parameter enable_output
  
  MSKrescodee trm;
  {
    auto r = msk.Optimize();
    msk.WriteData("__test.ptf");
    std::cout << "MosekSolver::Solve() optimize -> " << r << std::endl;
    if (! r.ok()) return r.status();
    trm = *r;
  }

  MSKsoltypee whichsol{};
  bool soldef = true;
  if      (msk.SolutionDef(MSK_SOL_ITG)) { whichsol = MSK_SOL_ITG; }
  else if (msk.SolutionDef(MSK_SOL_BAS)) { whichsol = MSK_SOL_BAS; }
  else if (msk.SolutionDef(MSK_SOL_ITR)) { whichsol = MSK_SOL_ITR; }
  else { soldef = false; }

  TerminationProto trmp;
  MSKprostae prosta{};
  MSKsolstae solsta{};
  if (! soldef) {
    auto [msg,name,code] = msk.LastError();
    trmp = TerminateForReason(msk.IsMaximize(), TerminationReasonProto::TERMINATION_REASON_NO_SOLUTION_FOUND, msg);
  }
  else {
    std::cout << "MosekSolver::Solve() solution is defined " << whichsol << std::endl;
    prosta = msk.GetProSta(whichsol);
    solsta = msk.GetSolSta(whichsol);

    // Attempt to determine TerminationProto from Mosek Termination code,
    // problem status and solution status.

    if      (solsta == MSK_SOL_STA_OPTIMAL ||
             solsta == MSK_SOL_STA_INTEGER_OPTIMAL) {
      std::cout << "MosekSolver::Solve() trmp = Optimal! " << std::endl;
      trmp = OptimalTerminationProto(msk.GetPrimalObj(whichsol),msk.GetDualObj(whichsol),"");
    }
    else if (solsta == MSK_SOL_STA_PRIM_INFEAS_CER) 
      trmp = InfeasibleTerminationProto(msk.IsMaximize(), FeasibilityStatusProto::FEASIBILITY_STATUS_FEASIBLE);
    else if (prosta == MSK_PRO_STA_PRIM_INFEAS_OR_UNBOUNDED)
      trmp = InfeasibleOrUnboundedTerminationProto(msk.IsMaximize());
    else if (solsta == MSK_SOL_STA_DUAL_INFEAS_CER)
      trmp = UnboundedTerminationProto(msk.IsMaximize());
    else if (solsta == MSK_SOL_STA_PRIM_AND_DUAL_FEAS ||
             solsta == MSK_SOL_STA_PRIM_FEAS) {
      LimitProto lim = LimitProto::LIMIT_UNSPECIFIED;
      switch (trm) {
        case MSK_RES_TRM_MAX_ITERATIONS: lim = LimitProto::LIMIT_ITERATION; break;
        case MSK_RES_TRM_MAX_TIME: lim = LimitProto::LIMIT_TIME; break;
        case MSK_RES_TRM_NUM_MAX_NUM_INT_SOLUTIONS: lim = LimitProto::LIMIT_SOLUTION; break;
#if MSK_VERSION_MAJOR >= 11
        case MSK_RES_TRM_SERVER_MAX_MEMORY: lim = LimitProto::LIMIT_MEMORY; break;
#endif
        // LIMIT_CUTOFF
        case MSK_RES_TRM_OBJECTIVE_RANGE: lim = LimitProto::LIMIT_OBJECTIVE; break;
        case MSK_RES_TRM_NUMERICAL_PROBLEM: lim = LimitProto::LIMIT_NORM; break;
        case MSK_RES_TRM_USER_CALLBACK: lim = LimitProto::LIMIT_INTERRUPTED; break;
        case MSK_RES_TRM_STALL: lim = LimitProto::LIMIT_SLOW_PROGRESS; break;
        default: lim = LimitProto::LIMIT_OTHER; break;
      }
      if      (solsta == MSK_SOL_STA_PRIM_AND_DUAL_FEAS)
        trmp = FeasibleTerminationProto(msk.IsMaximize(),lim,msk.GetPrimalObj(whichsol),msk.GetDualObj(whichsol));
      else 
        trmp = FeasibleTerminationProto(msk.IsMaximize(),lim,msk.GetPrimalObj(whichsol),std::nullopt);
    }
    else {
      trmp = NoSolutionFoundTerminationProto(msk.IsMaximize(), LimitProto::LIMIT_UNSPECIFIED);
    }
  }

  SolveResultProto result;
  *result.mutable_termination() = trmp;

  if (soldef) {
    std::cout << "MosekSolver::Solve() whichsol = " << whichsol << ", solsta =  " << solsta << std::endl;
    switch (solsta) { 
      case MSK_SOL_STA_OPTIMAL:
      case MSK_SOL_STA_INTEGER_OPTIMAL:
      case MSK_SOL_STA_PRIM_FEAS:
      case MSK_SOL_STA_DUAL_FEAS:
      case MSK_SOL_STA_PRIM_AND_DUAL_FEAS:
        {
          auto r = Solution(whichsol);
          std::cout << "MosekSolver::Solve() solution ok ? " << r.ok() << std::endl;
          if (r.ok()) {
            *result.add_solutions() = std::move(*r);
          }
        }
        break;
      case MSK_SOL_STA_DUAL_INFEAS_CER:
        {
          auto r = PrimalRay(whichsol);
          if (r.ok()) {
            *result.add_primal_rays() = std::move(*r);
          }
        }
        break;
      case MSK_SOL_STA_PRIM_INFEAS_CER:
        {
          auto r = DualRay(whichsol);
          if (r.ok()) {
            *result.add_dual_rays() = std::move(*r);
          }
        }
        break;
      case MSK_SOL_STA_PRIM_ILLPOSED_CER:
      case MSK_SOL_STA_DUAL_ILLPOSED_CER:
      case MSK_SOL_STA_UNKNOWN:
        break;
    }
  }
  return result;
}


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

absl::StatusOr<ComputeInfeasibleSubsystemResultProto>
MosekSolver::ComputeInfeasibleSubsystem(const SolveParametersProto&,
                                        MessageCallback,
                                        const SolveInterrupter*) {
  return absl::UnimplementedError(
      "HiGHS does not provide a method to compute an infeasible subsystem");
}

MATH_OPT_REGISTER_SOLVER(SOLVER_TYPE_MOSEK, MosekSolver::New);

} // namespace operations_research::math_opt

