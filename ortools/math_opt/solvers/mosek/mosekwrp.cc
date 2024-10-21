#include <mosek.h>
#include <cmath>
#include <limits>
#include <locale>
#include "absl/status/status.h"
#include "mosekwrp.cc"


namespace operations_research::math_opt {


  Mosek * Mosek::Create() {
    MSKtask_t task;
    MSKrescodee r = MSK_makeemptytask(nullptr,&task);
    if (r != MSK_RES_OK) {
      return nullptr;
    }

    return new Mosek(task);
  }

  Mosek::Mosek(MSKtask_t task) : task(task) { }
  Mosek::Mosek(Mosek && m) : task(std::move(m.task)) { }

  void Mosek::PutName(const std::string & name) {
    MSK_puttaskname(task.get(),name.c_str());
  }
  void Mosek::PutObjName(const std::string & name) {
    MSK_putobjname(task.get(),name.c_str());
  }
  void Mosek::PutVarName(VariableIndex j, const std::string & name) {
    MSK_putvarname(task.get(),j,name.c_str());
  }
  void Mosek::PutConName(ConstraintIndex i, const std::string & name) {
    MSK_putconname(task.get(),i,name.c_str());
  }
  void Mosek::PutObjectiveSense(bool maximize) {
    MSK_putobjsense(task.get(),maximize ? MSK_OBJECTIVE_SENSE_MAXIMIZE : MSK_OBJECTIVE_SENSE_MINIMIZE);
  }

  absl::StatusOr<VariableIndex> Mosek::AppendVars(const std::vector<double>& lb,
                                           const std::vector<double>& ub) {
    if (lb.size() != ub.size())
      return absl::InvalidArnumentError("Mismatching lengths of lb and ub");
    size_t n = lb.size();
    int numvar = NumVar();
    if (n > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("arguments lb and ub too large");
    
    MSK_appendvars(task.get(),(int)n);
    std::vector<MSKboundkeye> bk(n);
    for (ssize_t i = 0; i < n; ++i) {
      bk[i] = 
        ( lb[i] > ub[i] ? MSK_BK_RA 
          : ( std::isfinite(lb[i]) ? 
              ( std::isfinite(ub[i]) ? 
                ( lb[i] < ub[i] ? MSK_BK_RA : MSK_BK_FX )
                : MSK_BK_LO )
              : ( std::isfinite(ub[i] ? MSK_BK_UP : MSK_BK_FR) ) ) );
    }
    
    MSK_putvarboundslice(task.get(),(int)n,bk.data(),bl.data(),bu.data());
    return absl::OkStatus();
  }
  absl::StatusOr<ConstraintIndex> Mosek::AppendCons(const std::vector<double>& lb,
                                             const std::vector<double>& ub) {
    if (lb.size() != ub.size())
      return absl::InvalidArgumentError("Mismatching lengths of lb and ub");
    size_t n = lb.size();
    int numcon = NumCon();
    if (n > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("arguments lb and ub too large");
    
    MSK_appendcons(task.get(),(int)n);
    std::vector<MSKboundkeye> bk(n);
    for (ssize_t i = 0; i < n; ++i) {
      bk[i] = 
        ( lb[i] > ub[i] ? MSK_BK_RA 
          : ( std::isfinite(lb[i]) ? 
              ( std::isfinite(ub[i]) ? 
                ( lb[i] < ub[i] ? MSK_BK_RA : MSK_BK_FX )
                : MSK_BK_LO )
              : ( std::isfinite(ub[i] ? MSK_BK_UP : MSK_BK_FR) ) ) );
    }
    
    MSK_putconboundslice(task.get(),(int)n,bk.data(),bl.data(),bu.data());
    return absl::OkStatus();
  }
  absl::Status Mosek::PutVarType(VariableIndex j, bool is_integer) {
    if (MSK_RES_OK != MSK_putvartype(task.get(),j,is_integer ? MSK_VAR_TYPE_INT : MSK_VAR_TYPE_CONT))
      return absl::InvalidArgumentError("Arguments j is invalid");
    return absl::OkStatus();

  }

  absl::Status Mosek::PutC(const std::vector<double> & c) {
    auto n = c.size();
    if (n > NumVar())
      return absl::InvalidArgumentError("Argument c is too large");
    for (int i = 0; i < n; ++i)
      MSK_putcj(task.get(),j,c[j]);
    return absl::OkStatus();
  }

  void Mosek::PutCFix(double cfix) {
    MSK_putcfix(task.get(),cfix);
  }

  absl::Status Mosek::PutAIJList(const std::vector<ConstraintIndex>& subi,
                          const std::vector<VariableIndex>& subj,
                          const std::vector<double>& valij) {
    if (subi.size() != subj.size() ||
        subi.size() != valij.size())
      return absl::InvalidArgumentError("Mismatching arguments subi, subj, valij");
    size_t n = subi.size();
    if (n > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("Arguments subi, subj, valij are too long");

    if (MSK_RES_OK != MSK_putaijlist(task.get(),(int)n,subi.data(),subj.data(),valij.data()))
      return absl::InvalidArgumentError("Invalid index argument subi or subj");
    return absl::OkStatus();
  }

  absl::StatusOr<DisjunctiveConstraintIndex> Mosek::AppendIndicatorConstraint(
      bool negate,
      VariableIndex indvar, const std::vector<VariableIndex>& subj,
      const std::vector<double>& cof, double lb, double ub) {

    if (subj.size() != cof.size())
      return absl::InvalidArgumentError("Mismatching arguments subj, cof");

    size_t n = subj.size();
    if (n > std::numeric_limits<int>::max())
      return absl::InvalidArgumentError("Arguments subj or cof is too long");
    
    int64_t ndjc,nafe;
    MSK_getnumdjc(task.get(),&ndjc);
    MSK_getnumafe(task.get(),&nafe);

    MSK_appendafes(task.get(),2);
    MSK_appenddjcs(task.get(),1);

    MSK_putafefrowentry(task.get(),nafe, indvar,1.0);
    MSK_putafefrow(task.get(),nafe+1,(int)n,subj.data(),cof.data());
    int64_t afeidx[4] = { nafe, nafe, nafe+1, nafe+1 };
    double b[4] = { 0.0, 1.0, lb, ub };
      MSK_putdjc();
    }
   

    return ndjc;
  }
  absl::Status Mosek::PutDJCName(DisjunctiveConstraintIndex djci, const std::string & name);
  absl::Status Mosek::PutACCName(ConeConstraintIndex acci, const std::string & name);
  
  absl::StatusOr<ConeConstraintIndex> Mosek::AppendConeConstraint(
      ConeType ct, const std::vector<int64_t>& ptr,
      const std::vector<VariableIndex>& subj, const std::vector<double>& cof,
      const std::vector<double>& b);

  // Delete-ish
  absl::Status Mosek::ClearVariable(VariableIndex j);
  absl::Status Mosek::ClearConstraint(ConstraintIndex i);
  absl::Status Mosek::ClearConeConstraint(ConeConstraintIndex i);
  absl::Status Mosek::ClearDisjunctiveConstraint(DisjunctiveConstraintIndex i);

  // Update
  
  absl::Status Mosek::UpdateVariableLowerBound(VariableIndex j, double b);
  absl::Status Mosek::UpdateVariableUpperBound(VariableIndex j, double b);
  absl::Status Mosek::UpdateVariableType(VariableIndex j, bool is_integer);
  absl::Status Mosek::UpdateConstraintLowerBound(ConstraintIndex i, double b);
  absl::Status Mosek::UpdateConstraintUpperBound(ConstraintIndex i, double b);
  absl::Status Mosek::UpdateObjectiveSense(bool maximize);
  absl::Status Mosek::UpdateObjective(double fixterm,
                               const std::vector<VariableIndex>& subj,
                               const std::vector<double>& cof);
  absl::Status Mosek::UpdateA(const std::vector<ConstraintIndex> & subi, const std::vector<VariableIndex> & subj, const std::vector<double> & cof);

  void Mosek::PutParam(MSKiparame par, int value);
  void Mosek::PutParam(MSKdparame par, double value);
  // Query
  int Mosek::NumVar() const;
  int Mosek::NumCon() const;
  bool Mosek::IsMaximize() const;
  double Mosek::GetParam(MSKdparame dpar) const;
  int Mosek::GetParam(MSKiparame ipar) const;

  static void Mosek::delete_msk_task_func(MSKtask_t);

}

#endif
