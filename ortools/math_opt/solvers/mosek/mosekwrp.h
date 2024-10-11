#ifndef OR_TOOLS_MATH_OPT_SOLVERS_MOSEK_MOSEKWRP_H_
#define OR_TOOLS_MATH_OPT_SOLVERS_MOSEK_MOSEKWRP_H_

#include <memory>
#include <vector>
#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "mosek.h"

namespace operations_research::math_opt {

class Mosek {
  public:
    enum class ConeType {
      SecondOrderCone,
      RotatedSecondOrderCone
    };

    using VariableIndex = int32_t;
    using ConstraintIndex = int32_t;
    using DisjunctiveConstraintIndex = int64_t;
    using ConeConstraintIndex = int64_t;


    Mosek();
    Mosek(Mosek && m);

    void PutName(const std::string & name);
    void PutObjName(const std::string & name);
    absl::Status PutVarName(VariableIndex j, const std::string & name);
    absl::Status PutConName(ConstraintIndex j, const std::string & name);
    void PutObjectiveSense(bool maximize);

    absl::StatusOr<VariableIndex> AppendVars(const std::vector<double>& lb,
                                             const std::vector<double>& ub);
    absl::StatusOr<ConstraintIndex> AppendCons(const std::vector<double>& lb,
                                               const std::vector<double>& ub);
    absl::Status PutVarType(VariableIndex j, bool is_integer);

    absl::Status PutC(const std::vector<double> & c);
    absl::Status PutCFix(double cfix);

    absl::Status PutAIJList(const std::vector<ConstraintIndex>& subi,
                            const std::vector<VariableIndex>& subj,
                            const std::vector<double>& valij);

    absl::StatusOr<DisjunctiveConstraintIndex> AppendIndicatorConstraint(
        bool negate,
        VariableIndex indvar, const std::vector<VariableIndex>& subj,
        const std::vector<double>& cof, double lb, double ub);
    absl::Status PutDJCName(DisjunctiveConstraintIndex djci, const std::string & name);
    
    absl::StatusOr<ConeConstraintIndex> AppendConeConstraint(
        ConeType ct, const std::vector<int64_t>& ptr,
        const std::vector<VariableIndex>& subj, const std::vector<double>& cof,
        const std::vector<double>& b);

    // Delete-ish
    absl::Status ClearVariable(VariableIndex j);
    absl::Status ClearConstraint(ConstraintIndex i);
    absl::Status ClearConeConstraint(ConeConstraintIndex i);
    absl::Status ClearDisjunctiveConstraint(DisjunctiveConstraintIndex i);

    // Update
    
    absl::Status UpdateVariableLowerBound(VariableIndex j, double b);
    absl::Status UpdateVariableUpperBound(VariableIndex j, double b);
    absl::Status UpdateVariableType(VariableIndex j, bool is_integer);
    absl::Status UpdateConstraintLowerBound(ConstraintIndex i, double b);
    absl::Status UpdateConstraintUpperBound(ConstraintIndex i, double b);
    absl::Status UpdateObjectiveSense(bool maximize);
    absl::Status UpdateObjective(double fixterm,
                                 const std::vector<VariableIndex>& subj,
                                 const std::vector<double>& cof);
    // Query
    int NumVar() const;
    int NumCon() const;

  private:    
    static void delete_msk_task_func(MSKtask_t);

    std::unique_ptr<msktaskt,decltype(delete_msk_task_func)*> task;

};

}

#endif
