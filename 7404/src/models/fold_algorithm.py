import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple


@dataclass
class Clause:
    """Propositional clause: head(X) :- body_literals, naf_not_ab(optional)."""

    head: str
    body: List[str]  # literals like "a", "-b"
    naf_not: Optional[str] = None  # e.g. "ab0" meaning "not ab0(X)"

    def length(self) -> int:
        return len(self.body)


class FOLD:
    """
    FOLD Algorithm (Algorithm 3 in the paper) adapted to propositional literals.

    我们对齐 Algorithm 3 的控制流：
    - FOLD(E+,E-): sequential covering 生成 defaults' clauses D，以及异常/abnormal 子句 AB
    - SPECIALIZE: 用 FOIL 的 IG 选择 best literal；若 IG<=0 则进入 EXCEPTION
    - EXCEPTION: 递归学习异常规则并引入 ab 谓词，default 子句 body 加上 NAF: not ab(X)

    参考：`file://4163-Article Text-7217-1-10-20190705.pdf`（Algorithm 3 + IG 公式）
    """

    def __init__(self, max_rule_length: int = 5):
        self.max_rule_length = max_rule_length
        self.default_rules: List[Clause] = []
        self.ab_rules: List[Clause] = []
        self._ab_counter = 0
        # Algorithm 3: enumerate noisy samples as ground facts to ensure convergence
        # We model this as a memorized set of positive instance ids.
        self.noisy_pos_facts: Set[str] = set()
        # keep training sets for post-processing (Algorithm 3 discussion after pseudocode)
        self._train_E_plus: List[str] = []
        self._train_E_minus: List[str] = []
        # Post-processing thresholds (paper describes "significant number of negative examples")
        # We treat "significant" as a small increase in covered negatives, controlled by these parameters.
        self.neg_increase_abs_tol: int = 2
        self.neg_increase_frac_tol: float = 0.02  # 2% of |E-|

    # ---------- utilities ----------
    @staticmethod
    def _covers_clause(clause: Clause, inst_id: str, bk: Dict[str, Dict[str, int]], ab_true: Set[str]) -> bool:
        facts = bk.get(inst_id, {})
        # positive / classical negation literals
        for lit in clause.body:
            if lit not in facts:
                return False
        # NAF not ab(X)
        if clause.naf_not is not None and clause.naf_not in ab_true:
            return False
        return True

    def _derive_ab_true(self, inst_id: str, bk: Dict[str, Dict[str, int]]) -> Set[str]:
        """Forward chain abnormal predicates until fixpoint."""
        ab_true: Set[str] = set()
        changed = True
        while changed:
            changed = False
            for c in self.ab_rules:
                if c.head in ab_true:
                    continue
                # ab clause has no naf_not
                if self._covers_clause(Clause(head=c.head, body=c.body, naf_not=None), inst_id, bk, ab_true=set()):
                    ab_true.add(c.head)
                    changed = True
        return ab_true

    def _covered_set(self, clause: Clause, examples: Iterable[str], bk: Dict[str, Dict[str, int]]) -> Set[str]:
        covered: Set[str] = set()
        for inst_id in examples:
            ab_true = self._derive_ab_true(inst_id, bk)
            if self._covers_clause(clause, inst_id, bk, ab_true):
                covered.add(inst_id)
        return covered

    @staticmethod
    def _foil_ig(p0: int, n0: int, p1: int, n1: int) -> float:
        """
        IG(L,R) = t * ( log2(p1/(p1+n1)) - log2(p0/(p0+n0)) )
        where t is the number of positive examples covered by R and R+L together.
        In FOIL, t = p1.
        """
        if p1 <= 0:
            return 0.0
        if (p1 + n1) == 0 or (p0 + n0) == 0:
            return 0.0
        # avoid log(0)
        def safe_log2(x: float) -> float:
            return math.log(x, 2) if x > 0 else float("-inf")

        term1 = safe_log2(p1 / (p1 + n1))
        term0 = safe_log2(p0 / (p0 + n0))
        if term1 == float("-inf") or term0 == float("-inf"):
            return 0.0
        return p1 * (term1 - term0)

    def _candidate_literals(self, E_plus: Iterable[str], bk: Dict[str, Dict[str, int]]) -> Set[str]:
        """ρ(c): all literals appearing in current positive examples."""
        lits: Set[str] = set()
        for inst_id in E_plus:
            lits.update(bk.get(inst_id, {}).keys())
        return lits

    @staticmethod
    def _complement(lit: str) -> str:
        """classical negation complement: a <-> -a"""
        return lit[1:] if lit.startswith("-") else f"-{lit}"

    # ---------- Algorithm 3 core ----------
    def fit(self, target: str, E_plus: Iterable[str], E_minus: Iterable[str], bk: Dict[str, Dict[str, int]]):
        self.default_rules = []
        self.ab_rules = []
        self._ab_counter = 0
        self.noisy_pos_facts = set()

        E_plus = list(E_plus)
        E_minus = list(E_minus)
        self._train_E_plus = list(E_plus)
        self._train_E_minus = list(E_minus)

        # Algorithm 3: while |E+|>0
        while len(E_plus) > 0:
            c = Clause(head=target, body=[])  # target :- true.
            c_hat = self._specialize(c, E_plus, E_minus, bk)

            # E+ <- E+ \ covers(c_hat, E+, B)
            covered_pos = self._covered_set(c_hat, E_plus, bk)
            E_plus = [e for e in E_plus if e not in covered_pos]
            self.default_rules.append(c_hat)

        # Post-processing mentioned after Algorithm 3:
        # 1) eliminate redundant/counterproductive predicates inside clauses
        # 2) sort clauses by positives covered (ascending) and remove clauses
        #    whose elimination doesn't affect overall coverage of positives
        self._prune_clauses(bk, target)

    def _add_best_literal(self, c: Clause, E_plus: List[str], E_minus: List[str], bk: Dict[str, Dict[str, int]]) -> Tuple[Clause, float]:
        """
        Algorithm 1/3: ADD_BEST_LITERAL(c,E+,E-)
        returns best specialized clause c_def and its IG.
        """
        # current coverage
        cov_pos_0 = self._covered_set(c, E_plus, bk)
        cov_neg_0 = self._covered_set(c, E_minus, bk)
        p0, n0 = len(cov_pos_0), len(cov_neg_0)

        best_ig = 0.0
        best_lit: Optional[str] = None

        candidates = self._candidate_literals(cov_pos_0, bk)
        # Remove literals already in body, and remove complements of literals already in body
        body_set = set(c.body)
        candidates = {
            lit
            for lit in candidates
            if lit not in body_set and self._complement(lit) not in body_set
        }

        for lit in candidates:
            c1 = Clause(head=c.head, body=c.body + [lit], naf_not=c.naf_not)
            cov_pos_1 = self._covered_set(c1, E_plus, bk)
            cov_neg_1 = self._covered_set(c1, E_minus, bk)
            p1, n1 = len(cov_pos_1), len(cov_neg_1)
            ig = self._foil_ig(p0, n0, p1, n1)
            if ig > best_ig:
                best_ig = ig
                best_lit = lit

        if best_lit is None:
            return c, 0.0

        return Clause(head=c.head, body=c.body + [best_lit], naf_not=c.naf_not), best_ig

    def _specialize(self, c: Clause, E_plus: List[str], E_minus: List[str], bk: Dict[str, Dict[str, int]]) -> Clause:
        """
        Algorithm 3: SPECIALIZE(c, E+, E-)
        """
        # while |E-|>0 and c.length < max_rule_length
        while len(E_minus) > 0 and c.length() < self.max_rule_length:
            c_def, ig_hat = self._add_best_literal(c, E_plus, E_minus, bk)
            if ig_hat > 0:
                c = c_def
            else:
                # EXCEPTION(c, E-, E+)  (swap)
                c_exc = self._exception(c_def, E_minus, E_plus, bk)
                if c_exc is None:
                    # Algorithm 3 line 16-18: enumerate(c, E+)
                    # In the paper, noisy samples are added as ground facts so the hypothesis converges.
                    # Here we "memorize" one uncovered positive as a fact: target(inst)=True.
                    self._enumerate_noisy(E_plus)
                    return c
                c = c_exc

            # Align with FOIL Algorithm 1 inner-loop:
            # update E- to negatives still covered by current clause; keep E+ unchanged here.
            # This keeps clauses more general and reduces the number of clauses in sequential covering.
            covered_neg = self._covered_set(c, E_minus, bk)
            E_minus = [e for e in E_minus if e in covered_neg]

        return c

    def _exception(self, c_def: Clause, E_plus: List[str], E_minus: List[str], bk: Dict[str, Dict[str, int]]) -> Optional[Clause]:
        """
        Algorithm 3: EXCEPTION(c_def, E+, E-)
        Called as EXCEPTION(c, E-, E+), so here E_plus are the original negatives.
        """
        _, ig_hat = self._add_best_literal(c_def, E_plus, E_minus, bk)
        if ig_hat <= 0:
            return None

        # c_set <- FOLD(E+, E-)   (recursive)
        sub = FOLD(max_rule_length=self.max_rule_length)
        sub.fit(c_def.head, E_plus, E_minus, bk)

        ab_name = f"ab{self._ab_counter}"
        self._ab_counter += 1

        # AB <- AB ∪ { ab :- body(c) } for c in c_set
        for sc in sub.default_rules:
            self.ab_rules.append(Clause(head=ab_name, body=sc.body, naf_not=None))

        # return c_hat <- head(c_def) :- body(c_def), not ab
        return Clause(head=c_def.head, body=c_def.body, naf_not=ab_name)

    def _enumerate_noisy(self, E_plus: List[str]) -> None:
        """enumerate(c,E+): add (some) noisy positives as ground facts."""
        if not E_plus:
            return
        # Add a single example at a time (greedy) to mimic convergence guarantee.
        self.noisy_pos_facts.add(E_plus[0])

    def _prune_literals_in_clause(
        self,
        clause: Clause,
        E_plus_all: List[str],
        E_minus_all: List[str],
        bk: Dict[str, Dict[str, int]],
        neg_increase_tol: int,
        pos_drop_tol: int,
    ) -> Clause:
        """
        Remove redundant literals if dropping them does not:
        - decrease covered positives by more than pos_drop_tol
        - increase covered negatives by more than neg_increase_tol
        """
        if not clause.body:
            return clause

        cur = clause
        base_pos = self._covered_set(cur, E_plus_all, bk)
        base_neg = self._covered_set(cur, E_minus_all, bk)

        changed = True
        while changed and len(cur.body) > 1:
            changed = False
            for lit in list(cur.body):
                cand_body = [x for x in cur.body if x != lit]
                cand = Clause(head=cur.head, body=cand_body, naf_not=cur.naf_not)
                cand_pos = self._covered_set(cand, E_plus_all, bk)
                cand_neg = self._covered_set(cand, E_minus_all, bk)

                if (len(base_pos) - len(cand_pos)) <= pos_drop_tol and (len(cand_neg) - len(base_neg)) <= neg_increase_tol:
                    cur = cand
                    base_pos, base_neg = cand_pos, cand_neg
                    changed = True
                    break

        return cur

    def _prune_clauses(self, bk: Dict[str, Dict[str, int]], target: str) -> None:
        """Post-processing described after Algorithm 3."""
        # Use real training E+/E- collected in fit()
        E_plus_all = list(self._train_E_plus)
        E_minus_all = list(self._train_E_minus)

        # tolerance derived from |E-| (paper: "significant number of negative examples")
        neg_tol = max(self.neg_increase_abs_tol, int(len(E_minus_all) * self.neg_increase_frac_tol))

        pruned_defaults: List[Clause] = []
        for c in self.default_rules:
            pruned_defaults.append(
                self._prune_literals_in_clause(
                    c,
                    E_plus_all,
                    E_minus_all,
                    bk,
                    neg_increase_tol=neg_tol,
                    pos_drop_tol=0,
                )
            )

        # Clause elimination: sort by positives covered (ascending) and remove clauses
        # if elimination doesn't affect overall coverage of positives (paper discussion).
        # Coverage here is with respect to training positives.
        def cov_pos_size(cl: Clause) -> int:
            return len(self._covered_set(cl, E_plus_all, bk))

        pruned_defaults.sort(key=cov_pos_size)

        # compute full coverage
        def hypothesis_covered_pos(clauses: List[Clause]) -> Set[str]:
            covered: Set[str] = set(self.noisy_pos_facts)
            for cl in clauses:
                covered |= self._covered_set(cl, E_plus_all, bk)
            return covered

        full_cov = hypothesis_covered_pos(pruned_defaults)
        kept: List[Clause] = list(pruned_defaults)

        i = 0
        while i < len(kept):
            test = kept[:i] + kept[i + 1 :]
            if hypothesis_covered_pos(test) == full_cov:
                kept = test
                # do not increment i, re-check at same index
            else:
                i += 1

        self.default_rules = kept

    # ---------- inference ----------
    def predict(self, instance: str, bk: Dict[str, Dict[str, int]]) -> bool:
        # ground facts for noisy positives (Algorithm 3 enumerate)
        if instance in self.noisy_pos_facts:
            return True
        ab_true = self._derive_ab_true(instance, bk)
        for c in self.default_rules:
            if self._covers_clause(c, instance, bk, ab_true):
                return True
        return False

    def print_rules(self):
        if not self.default_rules and not self.ab_rules:
            print("No rules learned.")
            return

        print("Default clauses (D):")
        for i, c in enumerate(self.default_rules, start=1):
            body = ", ".join(c.body) if c.body else "true"
            naf = f", not {c.naf_not}" if c.naf_not else ""
            print(f"  d{i}: {c.head}(X) :- {body}{naf}.")

        if self.ab_rules:
            print("\nAbnormal clauses (AB):")
            for i, c in enumerate(self.ab_rules, start=1):
                body = ", ".join(c.body) if c.body else "true"
                print(f"  ab{i}: {c.head}(X) :- {body}.")