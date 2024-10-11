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
    using VariableIndex = int32_t;
    using ConstraintIndex = int32_t;
    using IndConstraintIndex = int64_t;


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

    absl::StatusOr<IndConstraintIndex> AppendIndicatorConstraint(
        VariableIndex j, const std::vector<VariableIndex>& subj,
        const std::vector<double>& cof, double lb, double ub);

        int NumVar() const;
    int NumCon() const;

  private:    
    static void delete_msk_task_func(MSKtask_t);

    std::unique_ptr<msktaskt,decltype(delete_msk_task_func)*> task;

};

}

#endif
