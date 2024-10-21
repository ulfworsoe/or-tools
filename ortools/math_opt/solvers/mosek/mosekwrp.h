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

    typedef int32_t VariableIndex              ;
    typedef int32_t ConstraintIndex            ;
    typedef int64_t DisjunctiveConstraintIndex ;
    typedef int64_t ConeConstraintIndex        ;

    Mosek(Mosek && m);

    static Mosek * Create();

    void PutName(const std::string & name);
    void PutObjName(const std::string & name);
    void PutVarName(VariableIndex j, const std::string & name);
    void PutConName(ConstraintIndex j, const std::string & name);
    void PutObjectiveSense(bool maximize);

    absl::StatusOr<VariableIndex> AppendVars(const std::vector<double>& lb,
                                             const std::vector<double>& ub);
    absl::StatusOr<ConstraintIndex> AppendCons(const std::vector<double>& lb,
                                               const std::vector<double>& ub);
    absl::Status PutVarType(VariableIndex j, bool is_integer);

    absl::Status PutC(const std::vector<double> & c);
    void PutCFix(double cfix);

    absl::Status PutAIJList(const std::vector<ConstraintIndex>& subi,
                            const std::vector<VariableIndex>& subj,
                            const std::vector<double>& valij);

    absl::StatusOr<DisjunctiveConstraintIndex> AppendIndicatorConstraint(
        bool negate,
        VariableIndex indvar, const std::vector<VariableIndex>& subj,
        const std::vector<double>& cof, double lb, double ub);
    absl::Status PutDJCName(DisjunctiveConstraintIndex djci, const std::string & name);
    absl::Status PutACCName(ConeConstraintIndex acci, const std::string & name);
    
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
    absl::Status UpdateA(const std::vector<ConstraintIndex> & subi, const std::vector<VariableIndex> & subj, const std::vector<double> & cof);

    void PutParam(MSKiparame par, int value);
    void PutParam(MSKdparame par, double value);
    // Query
    int NumVar() const;
    int NumCon() const;
    bool IsMaximize() const;
    double GetParam(MSKdparame dpar) const;
    int GetParam(MSKiparame ipar) const;

  private:    
    static void delete_msk_task_func(MSKtask_t);

    Mosek(MSKtask_t task);

    std::unique_ptr<msktaskt,decltype(delete_msk_task_func)*> task;

};

}

#endif
