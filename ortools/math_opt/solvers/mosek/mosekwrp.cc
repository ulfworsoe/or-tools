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
    MSKappendrzerodomain(0);

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

  // Note: We implement indicator constraints as a disjunctive constraint of the form:
  // [ indvar = (negate ? 1.0 : 0.0) ]
  //   OR 
  // [ indvar = (negate ? 0.0 : 1.0)
  //   lb <= Ax <= ub ]
  //
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
    int64_t dom_eq, dom_lb, dom_ub;
    MSK_appendrzerodomain(task.get(),1,&dom_eq);
    if (std::isfinite(lb) {
      MSK_appendrplusdomain(task.get(),1,&dom_lb);
    }
    else {
      MSK_appendrdomain(task.get(),1,&dom_lb);
    }
    if (std::isfinite(ub) {
      MSK_appendrminusdomain(task.get(),1,&dom_ub);
    }
    else {
      MSK_appendrdomain(task.get(),1,&dom_ub);
    }

    int64_t afeidx[4] = { nafe, nafe, nafe+1, nafe+1 };
    double b[4] = { negate ? 1.0 : 0.0, negate ? 0.0 : 1.0, lb, ub };
    int64_t domidxs[4] = { dom_eq, dom_eq, dom_lb,dom_ub };
    int64_t termsizes[2] = { 1,3 };
    MSK_putdjc(task.get(),ndjc,4,domidxs,4,afeidx,b,2,termsizes);
   
    return ndjc;
  }
  absl::Status Mosek::PutDJCName(DisjunctiveConstraintIndex djci, const std::string & name) {
    if (MSK_RES_OK != MSK_putdjcname(task.get(),djci,name.c_str()))
      return absl::InvalidArgumentError("Invalid argument djci");
    return absl::OkStatus();
  }
  absl::Status Mosek::PutACCName(ConeConstraintIndex acci, const std::string & name) {
    if (MSK_RES_OK != MSK_putaccname(task.get(),dcci,name.c_str()))
      return absl::InvalidArgumentError("Invalid argument acci");
    return absl::OkStatus();
  }
  
  absl::StatusOr<ConeConstraintIndex> Mosek::AppendConeConstraint(
      ConeType ct, const std::vector<int64_t>& sizes,
      const std::vector<VariableIndex>& subj, const std::vector<double>& cof,
      const std::vector<double>& b) {
    size_t n = sizes.size();
    size_t nnz = 0; for (auto & nz : sizes) nnz += nz;
    int64_t domidx;

    if (nnz != cof.size() ||
        nnz != subj.size())
      return absl::InvalidArgumentError("Mismatching argument lengths of subj and cof");
    if (n != b.size())
      return absl::InvalidArgumentError("Mismatching argument lengths b and ptr");

    switch (ct) {
      case MSK_CT_QUAD: MSK_appendquadraticconedomain(task.get(),n); break;
      default: return absl::InvalidArgumentError("Cone type not supported");
    }

    int64_t afei;
    MSK_getnumafe(task.get(),&afei);
    MSK_appendafes(task.get(),n);
   
    std::vector<int64_t> afeidxs(n); for (ssize_t i = 0; i < n; ++i) afeidxs[i] = afei+i;
    std::vector<int64_t> ptr(n); ptr[0] = 0; for (ssize_t i = 0; i < n; ++i) ptr[i+1] = ptr[i] + sizes[i];

    std::vector<double> accb(n);

    int64_t acci;
    MSK_getnumacc(task.get(),&acci);
    MSK_appendaccseq(task.get(),domidx,n,afei, accb.data());
    MSK_putafefrowlist(task.get(),n,afeidxs.data(),sizes.data(),ptr.data(),nnz,subj.data(),cof.data());
    for (ssize_t i = 0; i < n; ++i) 
      MSK_putafeglist(task.get(),afei+i,b[i]);
    return acci;
  }


  // Delete-ish
  absl::Status Mosek::ClearVariable(VariableIndex j) {
    if (MSK_RES_OK != MSK_putvarbound(task.get(),j,MSK_BK_FR,0.0,0.0))
      return absl::InvalidArgumentError("Invalid variable index j");
    return absl::OkStatus();
  }
  absl::Status Mosek::ClearConstraint(ConstraintIndex i) {
    if (MSK_RES_OK != MSK_putconbound(task.get(),i,MSK_BK_FR,0.0,0.0))
      return absl::InvalidArgumentError("Invalid constraint index i");
    int subj;
    double cof;
    MSK_putarow(task.get(),i,0,&subj,&cof);
    return absl::OkStatus();
  }
  absl::Status Mosek::ClearConeConstraint(ConeConstraintIndex i) {
    int64_t afeidxs;
    double b;
    if (MSK_RES_OK != MSKputacc(task.get(),i,0,&afeidxs,&b))
      return absl::InvalidArgumentError("Invalid constraint index i");
    return absl::OkStatus();
  }
  absl::Status Mosek::ClearDisjunctiveConstraint(DisjunctiveConstraintIndex i) {
    int64_t i64s;
    double f64s;
    if (MSK_RES_OK != MSKputdjc(task.get(),i,0,&i64s,0,&i64s,&f64s,0,i64s))
      return absl::InvalidArgumentError("Invalid constraint index i");
    return absl::OkStatus();
  }

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
