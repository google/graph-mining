-------------------------- MODULE AsynchronousUnionFindMachineCertifiedProof --------------------------
(****************************************************************************
Author: Siddhartha Jayanti

We present a machine-certified verification of the Jayanti-Tarjan Concurrent
Union Find Algorithm presented by Siddhartha Jayanti and Robert Tarjan in
[Jayanti and Tarjan, PODC 2016] https://dl.acm.org/doi/10.1145/2933057.2933108
[Jayanti and Tarjan, Distributed Computing 2021] https://link.springer.com/article/10.1007/s00446-020-00388-x

For the proof, we use the Meta-Configuration Tracking method developed by
Siddhartha Jayanti and Jayanti et al.
[Jayanti, MIT 2022] https://dspace.mit.edu/handle/1721.1/150219
[Jayanti et al., POPL 2024] https://dl.acm.org/doi/10.1145/3632924

A TLA Description of the Jayanti-Tarjan Concurrent Union Find Algorithm 

Pseudo-code of the algorithm is as follows:

DESCRIPTION:
    Code shown for process $p$
    
SHARED VARIABLES:
    - Par[1..N] is an array of "parent pointers"
LOCAL VARIABLES:
    - For each process p:
        x_p, y_p, u_p, v_p are local variables

ALGORITHM:
  F1:   Find_p(x_p):
            u_p <- x_p
  F2:       while (Par[u_p] != u_p): u_p <- Par[u_p]
  F3:       return u_p
        
  U1:   Unite_p(x_p, y_p):
            u_p <- x_p; v_p <- y_p
  U2:       while TRUE:
                if   u_p = v_p: goto U5
                elif u_p < v_p: if CAS(Par[u_p], u_p, v_p): goto U5
                elif u_p > v_p: if CAS(Par[v_p], v_p, u_p): goto U5
  U3:           while (Par[u_p] != u_p): u_p <- Par[u_p]
  U4:           while (Par[v_p] != v_p): v_p <- Par[v_p]
  U5:       return Ack
  
  Note: Each line has at most 1 shared memory instruction.  

TRANSLATION NOTES:
    - U3 and U4 replace calls to "u <- Find(u)" and "v <- Find(v)".
      We note that these calls can be replaced by the given lines of code.
    - We use an additional variable: For each process p: pc_p; here pc_p
      is interpreted as the program counter.

PROOF NOTES:
****************************************************************************)

EXTENDS Integers, FiniteSets, TLAPS, FiniteSetTheorems

CONSTANTS PROCSET, N, NIL, ACK

VARIABLES Par, x, y, u, v, pc, P
vars == <<Par, x, y, u, v, pc>>
augs == <<P>>
allvars == <<Par, x, y, u, v, pc, P>>
M == P

InvocationLines  == {"F1", "U1"}
Lines            == {"F1", "F2", "F3", "U1", "U2", "U3", "U4", "U5"}

NodeSet == 1..N
PowerSetNodes == SUBSET NodeSet

States      == [NodeSet -> PowerSetNodes]
Rets        == [PROCSET -> NodeSet \cup {NIL} \cup {ACK}]
AtomConfigs == [sigma: States, f: Rets] \* Set of all structures t, with t.State \in States and t.f \in Rets

Max(S) == CHOOSE X \in S : \A Y \in S : Y <= X

InitVars == /\ Par  = [z \in NodeSet |-> z]
            /\ x  \in [PROCSET -> NodeSet]
            /\ y  \in [PROCSET -> NodeSet]
            /\ u  \in [PROCSET -> NodeSet]
            /\ v  \in [PROCSET -> NodeSet]
            /\ pc \in [PROCSET -> InvocationLines]

sigmaInit   == [z \in NodeSet |-> {z}]
fInit       == [p \in PROCSET |-> NIL]
InitAug     == P = {[sigma |-> sigmaInit, f |-> fInit]}

Init == InitVars /\ InitAug


(*
  F1:   Find_p(x_p):
            u_p <- x_p
  F2:       while (Par[u_p] != u_p): u_p <- Par[u_p]
  F3:       return u_p
*)        
LineF1(p) == /\ pc[p] = "F1"
             /\ \E xnew \in NodeSet: 
                    /\ x' = [x EXCEPT ![p] = xnew]
                    /\ u' = [u EXCEPT ![p] = xnew]
             /\ pc' = [pc EXCEPT ![p] = "F2"]
             /\ UNCHANGED <<Par, y, v>>              

LineF2(p) == /\ pc[p] = "F2"
             /\ IF Par[u[p]] # u[p] 
                    THEN /\ u' = [u EXCEPT ![p] = Par[u[p]]]
                         /\ UNCHANGED <<Par, x, y, v, pc>> 
                    ELSE /\ pc' = [pc EXCEPT ![p] = "F3"]
                         /\ UNCHANGED <<Par, x, y, u, v>>        
                    
LineF3(p) == /\ pc[p] = "F3"
             /\ \E line \in InvocationLines: pc' = [pc EXCEPT ![p] = line]
             /\ UNCHANGED <<Par, x, y, u, v>>

(*
  U1:   Unite_p(x_p, y_p):
            u_p <- x_p; v_p <- y_p
  U2:       while TRUE:
                if   u_p = v_p: goto U5
                elif u_p < v_p: if CAS(Par[u_p], u_p, v_p): goto U5
                elif u_p > v_p: if CAS(Par[v_p], v_p, u_p): goto U5
  U3:           while (Par[u_p] != u_p): u_p <- Par[u_p]
  U4:           while (Par[v_p] != v_p): v_p <- Par[v_p]
  U5:       return Ack
*)  
LineU1(p) == /\ pc[p] = "U1"
             /\ \E xnew \in NodeSet:
                    /\ x' = [x EXCEPT ![p] = xnew]
                    /\ u' = [u EXCEPT ![p] = xnew]
             /\ \E ynew \in NodeSet:
                    /\ y' = [y EXCEPT ![p] = ynew]
                    /\ v' = [v EXCEPT ![p] = ynew]
             /\ pc' = [pc EXCEPT ![p] = "U2"]
             /\ UNCHANGED <<Par>>

LineU2(p) == /\ pc[p] = "U2"
             /\ CASE u[p] = v[p] -> (/\ pc' = [pc EXCEPT ![p] = "U5"]
                                     /\ UNCHANGED <<Par, x, y, u, v>>)
                  [] u[p] < v[p] -> (IF Par[u[p]] = u[p]
                                        THEN /\ Par' = [Par EXCEPT ![u[p]] = v[p]]
                                             /\ pc'  = [pc EXCEPT ![p] = "U5"]
                                             /\ UNCHANGED <<x, y, u, v>>
                                        ELSE /\ pc' = [pc EXCEPT ![p] = "U3"]
                                             /\ UNCHANGED <<Par, x, y, u, v>>)
                  [] OTHER      -> (IF Par[v[p]] = v[p]                             \* u[p] > v[p]
                                        THEN /\ Par' = [Par EXCEPT ![v[p]] = u[p]]
                                             /\ pc'  = [pc EXCEPT ![p] = "U5"]
                                             /\ UNCHANGED <<x, y, u, v>>  
                                        ELSE /\ pc' = [pc EXCEPT ![p] = "U3"]
                                             /\ UNCHANGED <<Par, x, y, u, v>>)
                    
LineU3(p) == /\ pc[p] = "U3"
             /\ IF Par[u[p]] # u[p] 
                    THEN /\ u' = [u EXCEPT ![p] = Par[u[p]]]
                         /\ UNCHANGED <<Par, x, y, v, pc>>
                    ELSE /\ pc' = [pc EXCEPT ![p] = "U4"]
                         /\ UNCHANGED <<Par, x, y, u, v>>
                
LineU4(p) == /\ pc[p] = "U4"
             /\ IF Par[v[p]] # v[p]
                    THEN /\ v' = [v EXCEPT ![p] = Par[v[p]]]
                         /\ UNCHANGED <<Par, x, y, u, pc>>
                    ELSE /\ pc' = [pc EXCEPT ![p] = "U2"]
                         /\ UNCHANGED <<Par, x, y, u, v>>
             
LineU5(p) == /\ pc[p] = "U5"
             /\ \E line \in InvocationLines: pc' = [pc EXCEPT ![p] = line]
             /\ UNCHANGED <<Par, x, y, u, v>>
             
(*** UF Augmenting Lines ***)


(*
  F1:   Find_p(x_p):
            u_p <- x_p
  F2:       while (Par[u_p] != u_p): u_p <- Par[u_p]
  F3:       return u_p
*)        
AugF1(p) == P' = P 

AugF2(p) == IF Par[u[p]] = u[p] 
                THEN P' = {t \in AtomConfigs : \E told \in P: /\ told.f[p] = NIL 
                                                              /\ t.sigma = told.sigma 
                                                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]}
                ELSE P' = P
                                                     
AugF3(p) == P' = {t \in AtomConfigs : \E told \in P: /\ told.f[p] = u[p]
                                                     /\ t.sigma = told.sigma
                                                     /\ t.f = [told.f EXCEPT ![p] = NIL]}

(*
  U1:   Unite_p(x_p, y_p):
            u_p <- x_p; v_p <- y_p
  U2:       while TRUE:
                if   u_p = v_p: goto U5
                elif u_p < v_p: if CAS(Par[u_p], u_p, v_p): goto U5
                elif u_p > v_p: if CAS(Par[v_p], v_p, u_p): goto U5
  U3:           while (Par[u_p] != u_p): u_p <- Par[u_p]
  U4:           while (Par[v_p] != v_p): v_p <- Par[v_p]
  U5:       return Ack
*)                                                       
AugU1(p) == P' = P

AugU2(p) == IF u[p] = v[p] 
                THEN P' = {t \in AtomConfigs : \E told \in P: /\ told.f[p] = NIL
                                                              /\ t.sigma = told.sigma
                                                              /\ t.f = [told.f EXCEPT ![p] = ACK]
                          }
                ELSE IF (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]) 
                        THEN P' = {t \in AtomConfigs : \E told \in P: /\ told.f[p] = NIL
                                                                      /\ \A z \in NodeSet: 
                                                                        (z \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                                                        (t.sigma[z] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                                                      /\ \A z \in NodeSet:
                                                                        (z \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                                                        (t.sigma[z] = told.sigma[z])
                                                                      /\ t.f = [told.f EXCEPT ![p] = ACK]
                                  }
                        ELSE P' = P

AugU3(p) == P' = P
AugU4(p) == P' = P

AugU5(p) == P' = {t \in AtomConfigs : \E told \in P: /\ told.f[p] = ACK
                                                     /\ t.sigma = told.sigma
                                                     /\ t.f = [told.f EXCEPT ![p] = NIL]}

ExecF1(p) == LineF1(p) /\ AugF1(p)                
ExecF2(p) == LineF2(p) /\ AugF2(p)                
ExecF3(p) == LineF3(p) /\ AugF3(p)
             
ExecU1(p) == LineU1(p) /\ AugU1(p)                
ExecU2(p) == LineU2(p) /\ AugU2(p)                
ExecU3(p) == LineU3(p) /\ AugU3(p)                
ExecU4(p) == LineU4(p) /\ AugU4(p)                
ExecU5(p) == LineU5(p) /\ AugU5(p)                

ExecStep(p) == \/ ExecF1(p)
               \/ ExecF2(p)
               \/ ExecF3(p)
               \/ ExecU1(p)
               \/ ExecU2(p)
               \/ ExecU3(p)
               \/ ExecU4(p)
               \/ ExecU5(p)

Next == \E p \in PROCSET: ExecStep(p)


(* Specification *)
Spec     == Init /\ [][Next]_allvars


(* Safety Properties: check with TLC Model using Spec *)
\* INVARIANTS
ValidPar    == Par \in [NodeSet -> NodeSet]
Validx      == x   \in [PROCSET -> NodeSet]
Validy      == y   \in [PROCSET -> NodeSet]
Validu      == u   \in [PROCSET -> NodeSet]
Validv      == v   \in [PROCSET -> NodeSet]
Validpc     == pc  \in [PROCSET -> Lines]
ValidP      == P   \in SUBSET AtomConfigs

TypeOK == /\ ValidPar
          /\ Validx
          /\ Validy
          /\ Validu
          /\ Validv
          /\ Validpc
          /\ ValidP

ParPointsUp       == \A z   \in NodeSet: Par[z] >= z

SigmaIsPartition1 == \A z \in NodeSet: \A t \in P: 
                        z \in t.sigma[z]
SigmaIsPartition2 == \A w, z \in NodeSet: \A t \in P: 
                        (w \in t.sigma[z]) => (t.sigma[w] = t.sigma[z])
SigmaIsCoarse     == \A w,z \in NodeSet: \A t \in P: 
                        (Par[w] = z) => (t.sigma[w] = t.sigma[z])
SigmaIsFine       == \A w,z \in NodeSet: \A t \in P: 
                        (w # z /\ Par[w] = w /\ Par[z] = z) => (t.sigma[w] # t.sigma[z])

InvF1U1           == \A p \in PROCSET: \A t \in P: (pc[p] \in InvocationLines) => (t.f[p] = NIL)
InvF2             == \A p \in PROCSET: \A t \in P: 
                        (pc[p] = "F2") => (t.sigma[u[p]] = t.sigma[x[p]] /\ t.f[p] = NIL)
InvF3             == \A p \in PROCSET: \A t \in P:
                        (pc[p] = "F3") => (t.f[p] = u[p])
InvU234           == \A p \in PROCSET: \A t \in P:
                        (pc[p] \in {"U2", "U3", "U4"}) => (t.sigma[u[p]] = t.sigma[x[p]] /\ t.sigma[v[p]] = t.sigma[y[p]] /\ t.f[p] = NIL)
InvU5             == \A p \in PROCSET: \A t \in P:
                        (pc[p] = "U5") => (t.f[p] = ACK)

Linearizable      == P # {}

I == /\ TypeOK
     /\ ParPointsUp
     /\ SigmaIsPartition1
     /\ SigmaIsPartition2
     /\ SigmaIsCoarse
     /\ SigmaIsFine
     /\ InvF1U1
     /\ InvF2
     /\ InvF3
     /\ InvU234
     /\ InvU5
     /\ Linearizable

(* Proof Assumptions *)
ASSUME NisNat == /\ N \in Nat
                 /\ N >= 1

ASSUME AckNilDef == /\ ACK \notin NodeSet
                    /\ NIL \notin NodeSet
                    /\ ACK # NIL

(* Basic Math *)
LEMMA MaxIntegers ==
  ASSUME NEW S \in SUBSET Int, S # {}, IsFiniteSet(S)
  PROVE  /\ Max(S) \in S
         /\ \A Y \in S : Y <= Max(S)
<1>. DEFINE Pred(T) == T \in SUBSET Int /\ T # {} => \E X \in T : \A Y \in T : Y <= X
<1>1. Pred({})
  OBVIOUS
<1>2. ASSUME NEW T, NEW X, Pred(T), X \notin T
      PROVE  Pred(T \cup {X})
  <2>. HAVE T \cup {X} \in SUBSET Int
  <2>1. CASE \A Y \in T : Y <= X
    BY <2>1, Isa
  <2>2. CASE \E Y \in T : ~(Y <= X)
    <3>. T # {}
      BY <2>2
    <3>1. PICK Y \in T : \A z \in T : z <= Y
      BY <1>2
    <3>2. X <= Y
      BY <2>2, <3>1
    <3>3. QED  BY <3>1, <3>2
  <2>. QED  BY <2>1, <2>2
<1>. HIDE DEF Pred
<1>3. Pred(S)  BY <1>1, <1>2, FS_Induction, IsaM("blast")
<1>. QED  BY <1>3, Zenon DEF Max, Pred


(* Basic facts about compressed tree representation *)
LEMMA MaxIsRoot == ASSUME TypeOK,
                          ParPointsUp,
                          SigmaIsPartition1,
                          SigmaIsPartition2,
                          SigmaIsCoarse,
                          SigmaIsFine,
                          NEW w \in NodeSet,
                          NEW t \in P,
                          (w = Par[w])
             PROVE Max(t.sigma[w]) = Par[Max(t.sigma[w])]
  <1> USE NisNat DEF TypeOK, ValidPar, ValidP, AtomConfigs, States, PowerSetNodes, NodeSet
  <1>1. DEFINE z == Max(t.sigma[w])
  <1>2. t.sigma[w] \in SUBSET NodeSet
    OBVIOUS  
  <1>3. IsFiniteSet(t.sigma[w])
    BY NisNat, <1>2, FS_Subset, FS_Interval
  <1>4. t.sigma[w] # {}
    BY DEF SigmaIsPartition1
  <1>5. z \in t.sigma[w]
    BY <1>3, <1>4, MaxIntegers
  <1>6. Par[z] >= z
    BY <1>5 DEF ParPointsUp
  <1>7. t.sigma[w] = t.sigma[z]
    BY <1>5 DEF SigmaIsPartition2
  <1>8. Par[z] \in NodeSet
    BY <1>5, <1>7 DEF SigmaIsCoarse, SigmaIsPartition1
  <1>9. Par[z] \in t.sigma[z]
    BY <1>5, <1>7 DEF SigmaIsPartition1, SigmaIsCoarse
  <1>10. Par[z] = z
    BY <1>6, <1>7, <1>8, <1>9 DEF Max, SigmaIsCoarse
  <1> QED     
    BY <1>10        
 
LEMMA UniqueRoot == ASSUME TypeOK,
                           SigmaIsPartition2,
                           SigmaIsFine,
                           NEW t \in P,
                           NEW w \in NodeSet,
                           NEW z \in NodeSet,
                           (w = Par[w] /\ z = Par[z] /\ z \in t.sigma[w])
             PROVE w = z
  <1> USE NisNat DEF TypeOK, ValidPar, ValidP, AtomConfigs, States, PowerSetNodes, NodeSet
  <1> QED
    BY DEF SigmaIsPartition2, SigmaIsFine

LEMMA RootIsMax == ASSUME TypeOK,
                          ParPointsUp,
                          SigmaIsPartition1,
                          SigmaIsPartition2,
                          SigmaIsCoarse,
                          SigmaIsFine,
                          NEW w \in NodeSet,
                          NEW t \in P,
                          (w = Par[w])
             PROVE Max(t.sigma[w]) = w
  <1> USE NisNat DEF TypeOK, ValidPar, ValidP, AtomConfigs, States, PowerSetNodes, NodeSet
  <1>1. DEFINE z == Max(t.sigma[w])
  <1>2. (w = Par[w] /\ Par[z] = z)
    BY MaxIsRoot  
  <1>3. IsFiniteSet(t.sigma[w])
    BY NisNat, <1>2, FS_Subset, FS_Interval
  <1>4. t.sigma[w] # {}
    BY DEF SigmaIsPartition1
  <1>5. z \in t.sigma[w]
    BY <1>3, <1>4, MaxIntegers
  <1>6. (w = Par[w] /\ Par[z] = z /\ z \in t.sigma[w])
    BY <1>2, <1>5
  <1> QED     
    BY <1>6, UniqueRoot        

(* Proof of Type Correctness *)
LEMMA InitTypeSafety == Init => TypeOK
  <1> USE DEF Init, InitVars, NodeSet
  <1> SUFFICES ASSUME Init
               PROVE  TypeOK
    OBVIOUS
  <1>1. ValidPar
    BY DEF ValidPar
  <1>2. Validx
    BY DEF Validx
  <1>3. Validy
    BY DEF Validy
  <1>4. Validu
    BY DEF Validu
  <1>5. Validv
    BY DEF Validv
  <1>6. Validpc
    BY DEF Validpc, InvocationLines, Lines
  <1>7. ValidP
    BY DEF ValidP, InitAug, sigmaInit, fInit, AtomConfigs, States, Rets, PowerSetNodes
  <1>8. QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7 DEF TypeOK

LEMMA NextTypeSafety == TypeOK /\ [Next]_allvars => TypeOK'
  <1> USE DEF TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, Next, allvars, ExecStep, NodeSet, Lines
  <1> SUFFICES ASSUME TypeOK,
                      [Next]_allvars
               PROVE  TypeOK'
    OBVIOUS
  <1>1. ASSUME NEW p \in PROCSET,
               ExecF1(p)
        PROVE  TypeOK'
    BY <1>1 DEF ExecF1, LineF1, AugF1
  <1>2. ASSUME NEW p \in PROCSET,
               ExecF2(p)
        PROVE  TypeOK'
   BY <1>2 DEF ExecF2, LineF2, AugF2
  <1>3. ASSUME NEW p \in PROCSET,
               ExecF3(p)
        PROVE  TypeOK'
   BY <1>3 DEF ExecF3, LineF3, AugF3, InvocationLines
  <1>4. ASSUME NEW p \in PROCSET,
               ExecU1(p)
        PROVE  TypeOK'
   BY <1>4 DEF ExecU1, LineU1, AugU1
  <1>5. ASSUME NEW p \in PROCSET,
               ExecU2(p)
        PROVE  TypeOK'
   BY <1>5 DEF ExecU2, LineU2, AugU2, InvocationLines
  <1>6. ASSUME NEW p \in PROCSET,
               ExecU3(p)
        PROVE  TypeOK'
   BY <1>6 DEF ExecU3, LineU3, AugU3
  <1>7. ASSUME NEW p \in PROCSET,
               ExecU4(p)
        PROVE  TypeOK'
   BY <1>7 DEF ExecU4, LineU4, AugU4
  <1>8. ASSUME NEW p \in PROCSET,
               ExecU5(p)
        PROVE  TypeOK'
   BY <1>8 DEF ExecU5, LineU5, AugU5, InvocationLines
  <1>9. CASE UNCHANGED allvars
   BY <1>9
  <1>10. QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9 DEF ExecStep, Next

THEOREM TypeSafety == Spec => []TypeOK
  <1> SUFFICES ASSUME Spec
               PROVE  []TypeOK
    OBVIOUS             
  <1> QED
    BY PTL, InitTypeSafety, NextTypeSafety DEF Spec
 
(* Proof of Linearizability *)

LEMMA InitI == Init => I
  <1> USE DEF Init, InitVars, InitAug
  <1> SUFFICES ASSUME Init
               PROVE  I
    OBVIOUS
  <1>1. TypeOK
    BY InitTypeSafety
  <1>2. ParPointsUp
    BY NisNat DEF ParPointsUp, NodeSet
  <1>3. SigmaIsPartition1
    BY DEF SigmaIsPartition1, sigmaInit
  <1>4. SigmaIsPartition2
    BY DEF SigmaIsPartition2, sigmaInit
  <1>5. SigmaIsCoarse
    BY DEF SigmaIsCoarse, sigmaInit
  <1>6. SigmaIsFine
    BY DEF SigmaIsFine, sigmaInit
  <1>7. InvF1U1
    BY DEF InvF1U1, fInit, InvocationLines, Lines
  <1>8. InvF2
    BY DEF InvF2, InvocationLines
  <1>9. InvF3
    BY DEF InvF3, InvocationLines
  <1>10. InvU234
    BY DEF InvU234, InvocationLines
  <1>11. InvU5
    BY DEF InvU5, InvocationLines
  <1>12. Linearizable
    BY DEF Linearizable, sigmaInit, fInit
  <1>13. QED
    BY <1>1, <1>10, <1>11, <1>12, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9 DEF I

LEMMA NextI == I /\ [Next]_allvars => I'
  <1> USE DEF I
  <1> SUFFICES ASSUME I,
                      [Next]_allvars
               PROVE  I'
    OBVIOUS
  <1>1. ASSUME NEW p \in PROCSET,
               ExecF1(p)
        PROVE  I'
    <2> USE DEF ExecF1, LineF1, AugF1
    <2>1. TypeOK'
      BY <1>1, NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>1 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>1 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      BY <1>1 DEF SigmaIsPartition2
    <2>5. SigmaIsCoarse'
      BY <1>1 DEF SigmaIsCoarse
    <2>6. SigmaIsFine'
      BY <1>1 DEF SigmaIsFine
    <2>7. InvF1U1'
      BY <1>1 DEF InvF1U1, InvocationLines
    <2>8. InvF2'
      BY <1>1 DEF TypeOK, Validx, Validu, InvF1U1, InvF2, InvocationLines
    <2>9. InvF3'
      BY <1>1 DEF InvF3
    <2>10. InvU234'
      BY <1>1 DEF InvU234
    <2>11. InvU5'
      BY <1>1 DEF InvU5
    <2>12. Linearizable'
      BY <1>1 DEF Linearizable
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>2. ASSUME NEW p \in PROCSET,
               ExecF2(p)
        PROVE  I'
    <2> USE <1>2 DEF ExecF2, LineF2, AugF2
    <2>1. TypeOK'
      BY NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>2 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      <3> USE DEF SigmaIsPartition1
      <3> SUFFICES ASSUME NEW z \in NodeSet',
                          NEW t \in P'
                   PROVE  (z \in t.sigma[z])'
        BY DEF SigmaIsPartition1
      <3>1. CASE Par[u[p]] = u[p]
        <4>1. PICK told \in P: /\ told.f[p] = NIL 
                                /\ t.sigma = told.sigma 
                                /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
          BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>4. SigmaIsPartition2'
      <3> USE DEF SigmaIsPartition2\*, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w \in t.sigma[z])'
                   PROVE  (t.sigma[w] = t.sigma[z])'
        BY DEF SigmaIsPartition2
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>5. SigmaIsCoarse'
      <3> USE DEF SigmaIsCoarse
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (Par[w] = z)'
                   PROVE  (t.sigma[w] = t.sigma[z])'
        BY DEF SigmaIsCoarse
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>6. SigmaIsFine'
      <3> USE DEF SigmaIsFine
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w # z /\ Par[w] = w /\ Par[z] = z)'
                   PROVE  (t.sigma[w] # t.sigma[z])'
        BY DEF SigmaIsFine
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>7. InvF1U1'
      <3> USE NextTypeSafety DEF InvF1U1, InvocationLines, TypeOK, Validpc
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in InvocationLines)'
                   PROVE  (t.f[p_1] = NIL)'
        BY DEF InvF1U1
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>8. InvF2'
      <3> USE DEF InvF2, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "F2")'
                   PROVE  (t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.f[p_1] = NIL)'
        BY DEF InvF2
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. t \in P
          BY <3>2
        <4>2. t.sigma[u[p_1]] = t.sigma[Par[u[p_1]]]
          BY <3>2, <4>1 DEF SigmaIsCoarse
        <4> QED
          BY <3>1, <4>1 DEF SigmaIsCoarse
      <3> QED
        BY <3>1, <3>2
    <2>9. InvF3'
      <3> USE NisNat, NextTypeSafety DEF InvF2, InvF3, InvocationLines, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "F3")'
                   PROVE  (t.f[p_1] = u[p_1])'
        BY DEF InvF3
       <3>1. CASE Par[u[p]] = u[p]
         <4> USE <3>1
         <4>a. u'[p] = u[p]
           OBVIOUS
         <4>b. u'[p] = Max(t.sigma[u'[p]])
           BY <2>1, <2>2, <2>3, <2>4, <2>5, <2>6, RootIsMax
         <4>1. PICK told \in P: /\ told.f[p] = NIL 
                                /\ t.sigma = told.sigma 
                                /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <4>1, <4>b
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. t \in P
          BY <3>2
        <4>2. t.sigma[u[p_1]] = t.sigma[Par[u[p_1]]]
          BY <3>2, <4>1 DEF SigmaIsCoarse
        <4> QED
          BY <3>1, <4>1 DEF SigmaIsCoarse
      <3> QED
        BY <3>1, <3>2
    <2>10. InvU234'
      <3> USE NextTypeSafety DEF InvU234, InvocationLines, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in {"U2", "U3", "U4"})'
                   PROVE  (t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]] /\ t.f[p_1] = NIL)'
        BY DEF InvU234
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>11. InvU5'
      <3> USE NextTypeSafety DEF InvU5, InvocationLines, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "U5")'
                   PROVE  (t.f[p_1] = ACK)'
        BY DEF InvU5
       <3>1. CASE Par[u[p]] = u[p]
         <4>1. \E told \in P: /\ told.f[p] = NIL 
                              /\ t.sigma = told.sigma 
                              /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
             BY <3>1
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>2
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>12. Linearizable'
       <3> USE NisNat DEF Linearizable, InvF2, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
       <3>1. PICK told \in P: TRUE
         BY Linearizable
       <3>a. told.f[p] = NIL
         BY <3>1
       <3>b. told.sigma[x[p]] \in PowerSetNodes
         OBVIOUS
       <3>c. told.sigma[x[p]] \in SUBSET Int
         OBVIOUS
       <3>d. x[p] \in NodeSet
         OBVIOUS
       <3>e. x[p] \in told.sigma[x[p]]
         BY DEF SigmaIsPartition1
       <3>f. told.sigma[x[p]] # {}
         BY <3>e
       <3>g. IsFiniteSet(told.sigma[x[p]])
         BY <3>b, FS_Subset, FS_Interval
       <3>h. Max(told.sigma[x[p]]) \in told.sigma[x[p]]
         BY <3>b, <3>c, <3>d, <3>e, <3>f, <3>g, MaxIntegers DEF SigmaIsPartition1
       <3>i. Max(told.sigma[x[p]]) \in 1..N
         BY <3>h DEF ValidP
       <3>j. [sigma |-> told.sigma, f |-> told.f] \in P
         BY <3>a
       <3>2 CASE Par[u[p]] = u[p]
        <4>1. [sigma |-> told.sigma, f |-> [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]] \in P'
          BY <3>1, <3>2, <3>a, <3>i, <3>j
        <4> QED
          BY <4>1
      <3>3 CASE Par[u[p]] # u[p]
        <4>1. P' = P 
          BY <3>3
        <4> QED
          BY <4>1           
      <3> QED
        BY <3>2, <3>3
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>3. ASSUME NEW p \in PROCSET,
               ExecF3(p)
        PROVE  I'
    <2> USE <1>3 DEF ExecF3, LineF3, AugF3
    <2>1. TypeOK'
      BY  NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>3 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>3 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      <3> USE DEF SigmaIsPartition2
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w \in t.sigma[z])'
                   PROVE  (t.sigma[w] = t.sigma[z])'
         BY DEF SigmaIsPartition2
       <3>1. PICK told \in P: /\ told.f[p] = u[p]
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>5. SigmaIsCoarse'
      <3> USE DEF SigmaIsCoarse
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (Par[w] = z)'
                   PROVE  (t.sigma[w] = t.sigma[z])'
         BY DEF SigmaIsCoarse
       <3>1. PICK told \in P: /\ told.f[p] = u[p]
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>6. SigmaIsFine'
      <3> USE DEF SigmaIsFine
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w # z /\ Par[w] = w /\ Par[z] = z)'
                   PROVE  (t.sigma[w] # t.sigma[z])'
         BY DEF SigmaIsFine
       <3>1. PICK told \in P: /\ told.f[p] = u[p]
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>7. InvF1U1'
      <3> USE DEF InvF1U1, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in InvocationLines)'
                   PROVE  (t.f[p_1] = NIL)'
         BY DEF InvF1U1
       <3>1. CASE p = p_1
         <4>1. PICK told \in P: /\ told.f[p] = u[p]
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = NIL]
             BY Zenon
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE p # p_1
         <4>1. PICK told \in P: /\ told.f[p] = u[p]
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = NIL]
             BY Zenon
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>8. InvF2'
      BY DEF InvF2, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>9. InvF3'
      BY DEF InvF3, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>10. InvU234'
      BY DEF InvU234, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>11. InvU5'
      BY DEF InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>12. Linearizable'
       <3> USE DEF Linearizable, InvF3, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
       <3>1. PICK told \in P: TRUE
         BY Linearizable
       <3>2. told.f[p] = u[p]
         OBVIOUS
       <3>3. [sigma |-> told.sigma, f |-> [told.f EXCEPT ![p] = NIL]] \in P'
        BY <3>1, <3>2 DEF AugU5
      <3> QED
        BY <3>3
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>4. ASSUME NEW p \in PROCSET,
               ExecU1(p)
        PROVE  I'
    <2> USE DEF ExecU1, LineU1, AugU1
    <2>1. TypeOK'
      BY <1>4, NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>4 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>4 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      BY <1>4 DEF SigmaIsPartition2
    <2>5. SigmaIsCoarse'
      BY <1>4 DEF SigmaIsCoarse
    <2>6. SigmaIsFine'
      BY <1>4 DEF SigmaIsFine
    <2>7. InvF1U1'
      BY <1>4 DEF InvF1U1, InvocationLines
    <2>8. InvF2'
      BY <1>4 DEF InvF2
    <2>9. InvF3'
      BY <1>4 DEF InvF3
    <2>10. InvU234'
      BY <1>4 DEF TypeOK, Validx, Validy, Validu, Validv, InvF1U1, InvU234, InvocationLines
    <2>11. InvU5'
      BY <1>4 DEF InvU5
    <2>12. Linearizable'
      BY <1>4 DEF Linearizable
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>5. ASSUME NEW p \in PROCSET,
               ExecU2(p)
        PROVE  I'
    <2> USE <1>5 DEF ExecU2, LineU2, AugU2, InvU234
    <2>1. TypeOK'
      BY <1>5, NextTypeSafety
    <2>2. ParPointsUp'
      BY NisNat DEF ParPointsUp, TypeOK, ValidPar, Validx, Validy, Validu, Validv, NodeSet
    <2>3. SigmaIsPartition1'
      <3> USE NisNat DEF SigmaIsPartition1, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW z \in NodeSet',
                          NEW t \in P'
                   PROVE  (z \in t.sigma[z])'
        BY DEF SigmaIsPartition1
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4> QED
          BY <4>1
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z] = told.sigma[z])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> QED
            BY <5>1
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5> QED
            OBVIOUS
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>4. SigmaIsPartition2'
      <3> USE NisNat DEF SigmaIsPartition2, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w \in t.sigma[z])'
                   PROVE  (t.sigma[w] = t.sigma[z])'
        BY DEF SigmaIsPartition2
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4> QED
          BY <4>1
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS   
          <5>a. DEFINE tsigx == t.sigma[x[p]]  
          <5>b. DEFINE tsigy == t.sigma[y[p]]
          <5>c. DEFINE tsigw == t.sigma[w]
          <5>d. DEFINE tsigz == t.sigma[z]
          <5>e. DEFINE toldsigx == told.sigma[x[p]]  
          <5>f. DEFINE toldsigy == told.sigma[y[p]]
          <5>g. DEFINE toldsigw == told.sigma[w]
          <5>h. DEFINE toldsigz == told.sigma[z]
          <5>2. CASE w \in toldsigx \cup toldsigy /\ z \in toldsigx \cup toldsigy
            <6> USE <5>1, <5>2
            <6>1. tsigw = toldsigx \cup toldsigy
              OBVIOUS
            <6>2. tsigz = toldsigx \cup toldsigy
              OBVIOUS
            <6> QED
              BY <6>1, <6>2
          <5>3. CASE w \in toldsigx \cup toldsigy /\ z \notin toldsigx \cup toldsigy
            <6> USE <5>1, <5>3
            <6>1. toldsigw = toldsigz
              OBVIOUS
            <6>2. toldsigw = toldsigx \/ toldsigw = toldsigy
              OBVIOUS
            <6>3. z \notin toldsigx /\ z \notin toldsigy
              OBVIOUS
            <6>4. z \in toldsigz
              BY DEF SigmaIsPartition1 
            <6>5. toldsigz # toldsigx /\ toldsigz # toldsigy
              BY <6>3, <6>4
            <6>6. toldsigw # toldsigz
              BY <6>2, <6>5
            <6>7. FALSE
              BY <6>1, <6>6
            <6> QED
              BY <6>7
          <5>4. CASE w \notin toldsigx \cup toldsigy /\ z \in toldsigx \cup toldsigy
            <6> USE <5>1, <5>4
            <6>1. toldsigw = toldsigz
              OBVIOUS
            <6>2. toldsigw # toldsigz
              BY DEF SigmaIsPartition1
            <6> QED
              BY <6>1, <6>2
          <5>5. CASE w \notin toldsigx \cup toldsigy /\ z \notin toldsigx \cup toldsigy
            <6> USE <5>1, <5>5
            <6>1. toldsigw = toldsigz
              OBVIOUS
            <6>2. tsigw = toldsigw
              OBVIOUS
            <6>3. tsigz = toldsigz
              OBVIOUS
            <6> QED
              BY <6>2, <6>3
          <5> QED
            BY <5>2, <5>3, <5>4, <5>5
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5> QED
            OBVIOUS
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>5. SigmaIsCoarse'
      <3> USE <1>5, NisNat DEF SigmaIsCoarse, SigmaIsPartition1, SigmaIsPartition2, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (Par[w] = z)'
                   PROVE  (t.sigma[w] = t.sigma[z])'
        BY DEF SigmaIsCoarse
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4> QED
          BY <4>1
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p])
          <5> USE <4>1
          <5>A. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> USE <5>A
          <5>a. DEFINE tsigx == t.sigma[x[p]]  
          <5>b. DEFINE tsigy == t.sigma[y[p]]
          <5>c. DEFINE tsigw == t.sigma[w]
          <5>d. DEFINE tsigz == t.sigma[z]
          <5>e. DEFINE toldsigx == told.sigma[x[p]]  
          <5>f. DEFINE toldsigy == told.sigma[y[p]]
          <5>g. DEFINE toldsigw == told.sigma[w]
          <5>h. DEFINE toldsigz == told.sigma[z]
          <5>1. CASE w = u[p]
            <6> USE <5>1
            <6>1. Par'[w] = v[p]
              OBVIOUS
            <6>2. w \in toldsigx
              OBVIOUS
            <6>3. w \in tsigx
              OBVIOUS
            <6>4. z = v[p]
              OBVIOUS
            <6>5. v[p] \in toldsigy
              BY <6>4
            <6>6. v[p] \in tsigy
              BY <6>5
            <6>7. tsigx = toldsigx \cup toldsigy
              OBVIOUS
            <6>8. tsigy = toldsigx \cup toldsigy
              OBVIOUS
            <6>9. tsigx = tsigy
              BY <6>7, <6>8
            <6> QED
              BY <6>5, <6>4, <6>6, <6>9
          <5>2. CASE w # u[p]
            <6> USE <5>2
            <6>1. Par'[w] = Par[w]
              OBVIOUS
            <6>2. z \in toldsigw
              OBVIOUS
            <6>3. (w \in toldsigx \cup toldsigy) => (z \in toldsigx \cup toldsigy)
              BY <6>1, <6>2
            <6>4. (w \notin toldsigx \cup toldsigy) => (z \notin toldsigx \cup toldsigy)
              BY <6>1, <6>2
            <6>5. CASE (w \in toldsigx \cup toldsigy)
              <7> USE <6>5
              <7>1 tsigw = toldsigx \cup toldsigy
                OBVIOUS
              <7>2 tsigz = toldsigx \cup toldsigy
                BY <6>3
              <7> QED 
                BY <7>1, <7>2
            <6>6. CASE (w \notin toldsigx \cup toldsigy)
              <7> USE <6>6
              <7>1 tsigw = toldsigw
                OBVIOUS
              <7>2 tsigz = toldsigz
                BY <6>4
              <7>3 tsigz = toldsigw
                BY <6>2, <7>1, <7>2
              <7> QED
                BY <7>1, <7>3
            <6> QED
              BY <6>5, <6>6
          <5> QED
            BY <5>1, <5>2
        <4>2. CASE (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>2
          <5>A. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> USE <5>A
          <5>a. DEFINE tsigx == t.sigma[x[p]]  
          <5>b. DEFINE tsigy == t.sigma[y[p]]
          <5>c. DEFINE tsigw == t.sigma[w]
          <5>d. DEFINE tsigz == t.sigma[z]
          <5>e. DEFINE toldsigx == told.sigma[x[p]]  
          <5>f. DEFINE toldsigy == told.sigma[y[p]]
          <5>g. DEFINE toldsigw == told.sigma[w]
          <5>h. DEFINE toldsigz == told.sigma[z]
          <5>1. CASE w = v[p]
            <6> USE <5>1
            <6>1. Par'[w] = u[p]
              OBVIOUS
            <6>2. w \in toldsigy
              OBVIOUS
            <6>3. x[p] \in toldsigx
              OBVIOUS
            <6>4. z = u[p]
              OBVIOUS
            <6>5. u[p] \in toldsigx
              BY <6>3
            <6>6. u[p] \in tsigx
              BY <6>5
            <6>7. tsigy = toldsigx \cup toldsigy
              OBVIOUS
            <6>8. tsigx = toldsigx \cup toldsigy
              OBVIOUS
            <6>9. tsigx = tsigy
              BY <6>7, <6>8
            <6> QED
              BY <6>5, <6>4, <6>6, <6>9
          <5>2. CASE w # v[p]
            <6> USE <5>2
            <6>1. Par'[w] = Par[w]
              OBVIOUS
            <6>2. z \in toldsigw
              BY <6>1 DEF SigmaIsCoarse
            <6>3. (w \in toldsigx \cup toldsigy) => (z \in toldsigx \cup toldsigy)
              BY <6>1, <6>2
            <6>4. (w \notin toldsigx \cup toldsigy) => (z \notin toldsigx \cup toldsigy)
              BY <6>1, <6>2
            <6>5. CASE (w \in toldsigx \cup toldsigy)
              <7> USE <6>5
              <7>1 tsigw = toldsigx \cup toldsigy
                OBVIOUS
              <7>2 tsigz = toldsigx \cup toldsigy
                BY <6>3
              <7> QED 
                BY <7>1, <7>2
            <6>6. CASE (w \notin toldsigx \cup toldsigy)
              <7> USE <6>6
              <7>1 tsigw = toldsigw
                OBVIOUS
              <7>2 tsigz = toldsigz
                BY <6>4
              <7>3 tsigz = toldsigw
                BY <6>2, <7>1, <7>2
              <7> QED
                BY <7>1, <7>3
            <6> QED
              BY <6>5, <6>6
          <5> QED
            BY <5>1, <5>2
        <4>3. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>3
          <5> QED
            OBVIOUS
        <4> QED
          BY <4>1, <4>2, <4>3
      <3> QED
        BY <3>1, <3>2       
    <2>6. SigmaIsFine'
      <3> USE NisNat DEF SigmaIsPartition1, SigmaIsPartition2, SigmaIsCoarse, SigmaIsFine, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w # z /\ Par[w] = w /\ Par[z] = z)'
                   PROVE  (t.sigma[w] # t.sigma[z])'
        BY DEF SigmaIsFine
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4> QED
          BY <4>1
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1, <2>1, <2>2, <2>3, <2>4, <2>5 DEF InvU234
          <5>A. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> USE <5>A
          <5>a. DEFINE tsigx == t.sigma[x[p]]  
          <5>b. DEFINE tsigy == t.sigma[y[p]]
          <5>c. DEFINE tsigw == t.sigma[w]
          <5>d. DEFINE tsigz == t.sigma[z]
          <5>e. DEFINE toldsigx == told.sigma[x[p]]  
          <5>f. DEFINE toldsigy == told.sigma[y[p]]
          <5>g. DEFINE toldsigw == told.sigma[w]
          <5>h. DEFINE toldsigz == told.sigma[z]
          <5>B. z \notin toldsigw /\ w \notin toldsigz
            OBVIOUS
          <5>1. CASE u[p] < v[p] /\ Par[u[p]] = u[p]
            <6> USE <5>1 
            <6>1. CASE w \in toldsigx \/ z \in toldsigx
              <7> USE <6>1
              <7>1. told.sigma[u[p]] = toldsigx
                OBVIOUS
              <7>2. u[p] \in toldsigx
                BY <7>1
              <7>3. Par[u[p]] = u[p]
                OBVIOUS
              <7>4. w = u[p] \/ z = u[p]
                BY UniqueRoot, <7>2, <7>3
              <7>5. Par'[u[p]] # u[p]
                OBVIOUS
              <7>6. Par'[w] # w \/ Par'[z] # z
                BY <7>4, <7>5
              <7>7 FALSE
                BY <7>6
              <7> QED
                BY <7>7
            <6>2. CASE w \in toldsigy /\ z \in toldsigy
              <7> USE <6>2
              <7>1. Par[w] = w /\ Par[z] = z
                OBVIOUS
              <7>2. w = z
                BY UniqueRoot, <7>1
              <7> QED
                BY <7>2
           <6>3 CASE w \in toldsigy /\ z \notin (toldsigx \cup toldsigy)
              <7> USE <6>3
              <7>1. tsigw = toldsigx \cup toldsigy
                OBVIOUS
              <7>2. tsigz = toldsigz
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2
              <7> QED
                BY <7>3
           <6>4 CASE z \in toldsigy /\ w \notin (toldsigx \cup toldsigy)
              <7> USE <6>4
              <7>1. tsigz = toldsigx \cup toldsigy
                OBVIOUS
              <7>2. tsigw = toldsigw
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2
              <7> QED
                BY <7>3
           <6>5 CASE w \notin (toldsigx \cup toldsigy) /\ z \notin (toldsigx \cup toldsigy)
              <7> USE <6>5
              <7>1. tsigw = toldsigw
                OBVIOUS
              <7>2. tsigz = toldsigz
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2, <5>B
              <7> QED
                BY <7>3
           <6> QED
             BY <6>1, <6>2, <6>3, <6>4, <6>5
         <5>2. CASE u[p] > v[p] /\ Par[v[p]] = v[p]
            <6> USE <5>2
            <6>1. CASE w \in toldsigy \/ z \in toldsigy
              <7> USE <6>1
              <7>1. told.sigma[v[p]] = toldsigy
                OBVIOUS
              <7>2. v[p] \in toldsigy
                BY <7>1
              <7>3. Par[v[p]] = v[p]
                OBVIOUS
              <7>4. w = v[p] \/ z = v[p]
                BY UniqueRoot, <7>2, <7>3
              <7>5. Par'[v[p]] # v[p]
                OBVIOUS
              <7>6. Par'[w] # w \/ Par'[z] # z
                BY <7>4, <7>5
              <7>7 FALSE
                BY <7>6
              <7> QED
                BY <7>7
            <6>2. CASE w \in toldsigx /\ z \in toldsigx
              <7> USE <6>2
              <7>1. Par[w] = w /\ Par[z] = z
                OBVIOUS
              <7>2. w = z
                BY UniqueRoot, <7>1
              <7> QED
                BY <7>2
           <6>3 CASE w \in toldsigx /\ z \notin (toldsigx \cup toldsigy)
              <7> USE <6>3
              <7>1. tsigw = toldsigx \cup toldsigy
                OBVIOUS
              <7>2. tsigz = toldsigz
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2
              <7> QED
                BY <7>3
           <6>4 CASE z \in toldsigx /\ w \notin (toldsigx \cup toldsigy)
              <7> USE <6>4
              <7>1. tsigz = toldsigx \cup toldsigy
                OBVIOUS
              <7>2. tsigw = toldsigw
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2
              <7> QED
                BY <7>3
           <6>5 CASE w \notin (toldsigx \cup toldsigy) /\ z \notin (toldsigx \cup toldsigy)
              <7> USE <6>5
              <7>1. tsigw = toldsigw
                OBVIOUS
              <7>2. tsigz = toldsigz
                OBVIOUS
              <7>3. tsigw # tsigz
                BY <7>1, <7>2, <5>B
              <7> QED
                BY <7>3
           <6> QED
             BY <6>1, <6>2, <6>3, <6>4, <6>5
         <5> QED
           BY <5>1, <5>2
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5> QED
            OBVIOUS
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>7. InvF1U1'
      <3> USE DEF InvF1U1, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in InvocationLines)'
                   PROVE  (t.f[p_1] = NIL)'
        BY DEF InvF1U1
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4>2. pc' = [pc EXCEPT ![p] = "U5"]
          OBVIOUS
        <4>3. t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>1
        <4>a. CASE p_1 = p
          <5> USE <4>a
          <5>1. pc'[p_1] = "U5"
            BY <4>2
          <5> QED
            BY <5>1
        <4>b. CASE p_1 # p
          <5> USE <4>b
          <5>1. pc'[p_1] = pc[p_1]
            BY <4>2
          <5>2. t.f[p_1] = told.f[p_1]
            BY <4>3
          <5> QED
            BY <5>1, <5>2
        <4> QED
          BY <4>a, <4>b
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5>2. pc' = [pc EXCEPT ![p] = "U5"]
            OBVIOUS
          <5>3. t.f = [told.f EXCEPT ![p] = ACK]
            BY <5>1
          <5> QED
            BY <5>2, <5>3
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. pc' = [pc EXCEPT ![p] = "U3"]
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>8. InvF2'
      <3> USE DEF InvF2, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "F2")'
                   PROVE  (t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.f[p_1] = NIL)'
        BY DEF InvF2
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4>2. pc' = [pc EXCEPT ![p] = "U5"]
          OBVIOUS
        <4>3. t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>1
        <4>4. t.sigma = told.sigma
          BY <4>1
        <4>a. CASE p_1 = p
          <5> USE <4>a
          <5>1. pc'[p_1] = "U5"
            BY <4>2
          <5> QED
            BY <5>1
        <4>b. CASE p_1 # p
          <5> USE <4>b
          <5>1. pc'[p_1] = pc[p_1]
            BY <4>2
          <5>2. t.f[p_1] = told.f[p_1]
            BY <4>3
          <5>3. t.sigma = told.sigma
            BY <4>4
          <5> QED
            BY <5>1, <5>2, <5>3
        <4> QED
          BY <4>a, <4>b
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> USE <5>1
          <5>2. pc' = [pc EXCEPT ![p] = "U5"]
            OBVIOUS
          <5>3. t.f = [told.f EXCEPT ![p] = ACK]
            BY <5>1
          <5>a. CASE x[p_1] \in told.sigma[x[p]] \cup told.sigma[y[p]]
            <6> USE <5>a
            <6>1. t.sigma[x[p_1]] = told.sigma[x[p]] \cup told.sigma[y[p]]
              OBVIOUS
            <6>2. pc[p_1] = "F2" => told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3. pc[p_1] = "F2" => u[p_1] \in told.sigma[x[p]] \cup told.sigma[y[p]]
              BY <6>2 DEF SigmaIsPartition1, SigmaIsPartition2
            <6>4. pc[p_1] = "F2" => t.sigma[u[p_1]] = told.sigma[x[p]] \cup told.sigma[y[p]]
              BY <6>3
            <6>5. pc[p_1] = "F2" => t.sigma[u[p_1]] = t.sigma[x[p_1]]
              BY <6>1, <6>4
            <6> QED
              BY <5>2, <5>3, <6>5
          <5>b. CASE x[p_1] \notin told.sigma[x[p]] \cup told.sigma[y[p]]
            <6> USE <5>b
            <6>1. t.sigma[x[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>2. pc[p_1] = "F2" => told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3. pc[p_1] = "F2" => t.sigma[u[p_1]] = told.sigma[x[p_1]]
              BY <6>1, <6>2 DEF SigmaIsPartition1, SigmaIsPartition2
            <6>5. pc[p_1] = "F2" => t.sigma[u[p_1]] = t.sigma[x[p_1]]
              BY <6>1, <6>3
            <6> QED
              BY <5>2, <5>3, <6>5
          <5> QED
            BY <5>a, <5>b
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. pc' = [pc EXCEPT ![p] = "U3"]
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>9. InvF3'
      <3> USE DEF InvF3, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "F3")'
                   PROVE  (t.f[p_1] = u[p_1])'
        BY DEF InvF3
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4>2. pc' = [pc EXCEPT ![p] = "U5"]
          OBVIOUS
        <4>3. t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>1
        <4>a. CASE p_1 = p
          <5> USE <4>a
          <5>1. pc'[p_1] = "U5"
            BY <4>2
          <5> QED
            BY <5>1
        <4>b. CASE p_1 # p
          <5> USE <4>b
          <5>1. pc'[p_1] = pc[p_1]
            BY <4>2
          <5>2. t.f[p_1] = told.f[p_1]
            BY <4>3
          <5> QED
            BY <5>1, <5>2
        <4> QED
          BY <4>a, <4>b
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5>2. pc' = [pc EXCEPT ![p] = "U5"]
            OBVIOUS
          <5>3. t.f = [told.f EXCEPT ![p] = ACK]
            BY <5>1
          <5> QED
            BY <5>2, <5>3
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. pc' = [pc EXCEPT ![p] = "U3"]
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>10. InvU234'
      <3> USE DEF TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in {"U2", "U3", "U4"})'
                   PROVE  (t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]] /\ t.f[p_1] = NIL)'
        BY DEF InvU234
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4>2. pc' = [pc EXCEPT ![p] = "U5"]
          OBVIOUS
        <4>3. t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>1
        <4>4. t.sigma = told.sigma
          BY <4>1
        <4>a. CASE p_1 = p
          <5> USE <4>a
          <5>1. pc'[p_1] = "U5"
            BY <4>2
          <5> QED
            BY <5>1
        <4>b. CASE p_1 # p
          <5> USE <4>b
          <5>1. pc'[p_1] = pc[p_1]
            BY <4>2
          <5>2. t.f[p_1] = told.f[p_1]
            BY <4>3
          <5>3. t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]]
            BY <4>4
          <5> QED
            BY <5>1, <5>2, <5>3
        <4> QED
          BY <4>a, <4>b
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1 DEF SigmaIsPartition1, SigmaIsPartition2
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5> USE <5>1
          <5>2. pc' = [pc EXCEPT ![p] = "U5"]
            OBVIOUS
          <5>3. t.f = [told.f EXCEPT ![p] = ACK]
            BY <5>1
          <5>d1. DEFINE toldsigx == told.sigma[x[p]]
          <5>d2. DEFINE toldsigy == told.sigma[y[p]]
          <5>a. CASE x[p_1] \in toldsigx \cup toldsigy /\ y[p_1] \in toldsigx \cup toldsigy
            <6> USE <5>a
            <6>1. t.sigma[x[p_1]] = toldsigx \cup toldsigy
              OBVIOUS
            <6>2. told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3. t.sigma[u[p_1]] = toldsigx \cup toldsigy
              BY <6>1, <6>2
            <6>4. t.sigma[y[p_1]] = toldsigx \cup toldsigy
              OBVIOUS
            <6>5. told.sigma[v[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>6. t.sigma[v[p_1]] = toldsigx \cup toldsigy
              BY <6>4, <6>5
            <6>7. t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]]
              BY <6>1, <6>3, <6>4, <6>6
            <6> QED
              BY <5>2, <5>3, <6>7
          <5>b. CASE x[p_1] \in toldsigx \cup toldsigy /\ y[p_1] \notin toldsigx \cup toldsigy
            <6> USE <5>b
            <6>1. t.sigma[x[p_1]] = toldsigx \cup toldsigy
              OBVIOUS
            <6>2. told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3. t.sigma[u[p_1]] = toldsigx \cup toldsigy
              BY <6>1, <6>2
            <6>4. t.sigma[y[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>5. told.sigma[v[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>6. t.sigma[v[p_1]] = told.sigma[y[p_1]]
              BY <6>4, <6>5
            <6>7. t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]]
              BY <6>1, <6>3, <6>4, <6>6
            <6> QED
              BY <5>2, <5>3, <6>7
          <5>c. CASE x[p_1] \notin toldsigx \cup toldsigy /\ y[p_1] \in toldsigx \cup toldsigy
            <6> USE <5>c
            <6>1. t.sigma[x[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>2. told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3a. u[p_1] \notin toldsigx \cup toldsigy
              OBVIOUS
            <6>3. t.sigma[u[p_1]] = told.sigma[x[p_1]]
              BY <6>1, <6>2, <6>3a
            <6>4. t.sigma[y[p_1]] = toldsigx \cup toldsigy
              OBVIOUS
            <6>5. told.sigma[v[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>6. t.sigma[v[p_1]] = toldsigx \cup toldsigy
              BY <6>4, <6>5
            <6>7. t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]]
              BY <6>1, <6>3, <6>4, <6>6
            <6> QED
              BY <5>2, <5>3, <6>7
          <5>d. CASE x[p_1] \notin toldsigx \cup toldsigy /\ y[p_1] \notin toldsigx \cup toldsigy
            <6> USE <5>d
            <6>1. t.sigma[x[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>2. told.sigma[u[p_1]] = told.sigma[x[p_1]]
              OBVIOUS
            <6>3a. u[p_1] \in told.sigma[x[p_1]]
              BY <6>2
            <6>3b. u[p_1] \notin toldsigx \cup toldsigy
              BY <6>3a
            <6>3. t.sigma[u[p_1]] = told.sigma[x[p_1]]
              BY <6>1, <6>2, <6>3a, <6>3b
            <6>4. t.sigma[y[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>5. told.sigma[v[p_1]] = told.sigma[y[p_1]]
              OBVIOUS
            <6>6. t.sigma[v[p_1]] = told.sigma[y[p_1]]
              BY <6>4, <6>5
            <6>7. t.sigma[u[p_1]] = t.sigma[x[p_1]] /\ t.sigma[v[p_1]] = t.sigma[y[p_1]]
              BY <6>1, <6>3, <6>4, <6>6
            <6> QED
              BY <5>2, <5>3, <6>7
          <5> QED
            BY <5>a, <5>b, <5>c, <5>d
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. pc' = [pc EXCEPT ![p] = "U3"]
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>11. InvU5'
      <3> USE DEF InvU5, TypeOK, ValidPar, Validx, Validy, Validu, Validv, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] = "U5")'
                   PROVE  (t.f[p_1] = ACK)'
        BY DEF InvU5
      <3>1. CASE u[p] = v[p]
         <4> USE <3>1
         <4>1. PICK told \in P: /\ told.f[p] = NIL
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = ACK]
             OBVIOUS
        <4>2. pc' = [pc EXCEPT ![p] = "U5"]
          OBVIOUS
        <4>3. t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>1
        <4>a. CASE p_1 = p
          <5> USE <4>a
          <5>1. pc'[p_1] = "U5"
            BY <4>2
          <5>2. t.f[p_1] = ACK
            BY <4>3
          <5> QED
            BY <5>1, <5>2
        <4>b. CASE p_1 # p
          <5> USE <4>b
          <5>1. pc'[p_1] = pc[p_1]
            BY <4>2
          <5>2. t.f[p_1] = told.f[p_1]
            BY <4>3
          <5> QED
            BY <5>1, <5>2
        <4> QED
          BY <4>a, <4>b
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. PICK told \in P: /\ told.f[p] = NIL
                                 /\ \A z_1 \in NodeSet: 
                                    (z_1 \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                    (t.sigma[z_1] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                                 /\ \A z_1 \in NodeSet:
                                    (z_1 \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                    (t.sigma[z_1] = told.sigma[z_1])
                                 /\ t.f = [told.f EXCEPT ![p] = ACK]
            OBVIOUS
          <5>2. pc' = [pc EXCEPT ![p] = "U5"]
            OBVIOUS
          <5>3. t.f = [told.f EXCEPT ![p] = ACK]
            BY <5>1
          <5> QED
            BY <5>2, <5>3
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. pc' = [pc EXCEPT ![p] = "U3"]
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>12. Linearizable'
      <3> USE DEF Linearizable, InvU234, TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, AtomConfigs, Rets, InvocationLines, States, NodeSet, PowerSetNodes
      <3>d. PICK told \in P: TRUE
        OBVIOUS
      <3> USE <3>d
      <3>1. CASE u[p] = v[p]
        <4> USE <3>1
        <4>1. told.f[p] = NIL
          OBVIOUS 
        <4>2. told = [sigma |-> told.sigma, f |-> told.f]
          BY <3>d DEF ValidP
        <4>3. DEFINE t == [sigma |-> told.sigma, f |-> [told.f EXCEPT ![p] = ACK]]
        <4>4. /\ told.f[p] = NIL
              /\ t.sigma = told.sigma
              /\ t.f = [told.f EXCEPT ![p] = ACK]
          BY <4>2
        <4>5. t \in AtomConfigs
          OBVIOUS
        <4>6 t \in P'
          BY <3>d, <4>2, <4>4, <4>5
        <4> QED
          BY <4>6
      <3>2. CASE u[p] # v[p]
        <4> USE <3>2
        <4>1. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
          <5> USE <4>1
          <5>1. told.f[p] = NIL
            OBVIOUS
          <5>2. told = [sigma |-> told.sigma, f |-> told.f]
            OBVIOUS
          <5>3. DEFINE tsig == [z \in NodeSet |-> IF z \in told.sigma[x[p]] \cup told.sigma[y[p]]
                                                     THEN told.sigma[x[p]] \cup told.sigma[y[p]]
                                                     ELSE told.sigma[z]]
          <5>4. DEFINE tf   == [told.f EXCEPT ![p] = ACK]
          <5>5. [sigma |-> tsig, f |-> tf] \in P'
            BY <5>1, <5>2                                        
          <5> QED
            BY <5>5
        <4>2. CASE ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
          <5> USE <4>2
          <5>1. told \in P
            OBVIOUS
          <5> QED
            BY <5>1
        <4> QED
          BY <4>1, <4>2
      <3> QED
        BY <3>1, <3>2       
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>6. ASSUME NEW p \in PROCSET,
               ExecU3(p)
        PROVE  I'
    <2> USE DEF ExecU3, LineU3, AugU3
    <2>1. TypeOK'
      BY <1>6, NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>6 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>6 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      BY <1>6 DEF SigmaIsPartition2
    <2>5. SigmaIsCoarse'
      BY <1>6 DEF SigmaIsCoarse
    <2>6. SigmaIsFine'
      BY <1>6 DEF SigmaIsFine
    <2>7. InvF1U1'
      BY <1>6 DEF InvF1U1, InvocationLines, TypeOK, Validpc
    <2>8. InvF2'
      BY <1>6 DEF InvF2, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>9. InvF3'
      BY <1>6 DEF InvF3, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>10. InvU234'
      BY <1>6, NextTypeSafety, AckNilDef DEF TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, SigmaIsCoarse, InvU234
    <2>11. InvU5'
      BY <1>6 DEF InvU5, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>12. Linearizable'
      BY <1>6 DEF Linearizable
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>7. ASSUME NEW p \in PROCSET,
               ExecU4(p)
        PROVE  I'
    <2> USE DEF ExecU4, LineU4, AugU4
    <2>1. TypeOK'
      BY <1>7, NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>7 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>7 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      BY <1>7 DEF SigmaIsPartition2
    <2>5. SigmaIsCoarse'
      BY <1>7 DEF SigmaIsCoarse
    <2>6. SigmaIsFine'
      BY <1>7 DEF SigmaIsFine
    <2>7. InvF1U1'
      BY <1>7 DEF InvF1U1, InvocationLines, TypeOK, Validpc
    <2>8. InvF2'
      BY <1>7 DEF InvF2, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>9. InvF3'
      BY <1>7 DEF InvF3, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>10. InvU234'
      BY <1>7, NextTypeSafety DEF TypeOK, ValidPar, Validx, Validy, Validu, Validv, Validpc, ValidP, SigmaIsCoarse, InvU234
    <2>11. InvU5'
      BY <1>7 DEF InvU5, InvocationLines, TypeOK, Validpc, Validx, Validu
    <2>12. Linearizable'
      BY <1>7 DEF Linearizable
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I
    
  <1>8. ASSUME NEW p \in PROCSET,
               ExecU5(p)
        PROVE  I'
    <2> USE <1>8 DEF ExecU5, LineU5, AugU5
    <2>1. TypeOK'
      BY <1>8, NextTypeSafety
    <2>2. ParPointsUp'
      BY <1>8 DEF ParPointsUp
    <2>3. SigmaIsPartition1'
      BY <1>8 DEF SigmaIsPartition1
    <2>4. SigmaIsPartition2'
      <3> USE DEF SigmaIsPartition2
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w \in t.sigma[z])'
                   PROVE  (t.sigma[w] = t.sigma[z])'
         BY DEF SigmaIsPartition2
       <3>1. PICK told \in P: /\ told.f[p] = ACK
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>5. SigmaIsCoarse'
      <3> USE DEF SigmaIsCoarse
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (Par[w] = z)'
                   PROVE  (t.sigma[w] = t.sigma[z])'
         BY DEF SigmaIsCoarse
       <3>1. PICK told \in P: /\ told.f[p] = ACK
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>6. SigmaIsFine'
      <3> USE DEF SigmaIsFine
      <3> SUFFICES ASSUME NEW w \in NodeSet', NEW z \in NodeSet',
                          NEW t \in P',
                          (w # z /\ Par[w] = w /\ Par[z] = z)'
                   PROVE  (t.sigma[w] # t.sigma[z])'
         BY DEF SigmaIsFine
       <3>1. PICK told \in P: /\ told.f[p] = ACK
                              /\ t.sigma = told.sigma
                              /\ t.f = [told.f EXCEPT ![p] = NIL]
         BY Zenon
      <3> QED
        BY <3>1
    <2>7. InvF1U1'
      <3> USE DEF InvF1U1, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets
      <3> SUFFICES ASSUME NEW p_1 \in PROCSET',
                          NEW t \in P',
                          (pc[p_1] \in InvocationLines)'
                   PROVE  (t.f[p_1] = NIL)'
         BY DEF InvF1U1
       <3>1. CASE p = p_1
         <4>1. PICK told \in P: /\ told.f[p] = ACK
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = NIL]
             BY Zenon
        <4> QED
          BY <3>1, <4>1
      <3>2. CASE p # p_1
         <4>1. PICK told \in P: /\ told.f[p] = ACK
                                /\ t.sigma = told.sigma
                                /\ t.f = [told.f EXCEPT ![p] = NIL]
             BY Zenon
        <4> QED
          BY <3>1, <4>1
      <3> QED
        BY <3>1, <3>2
    <2>8. InvF2'
      BY DEF InvF2, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>9. InvF3'
      BY DEF InvF3, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>10. InvU234'
      BY DEF InvU234, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>11. InvU5'
      BY DEF InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
    <2>12. Linearizable'
       <3> USE DEF Linearizable, InvU5, TypeOK, Validpc, Validpc, ValidP, AtomConfigs, Rets, InvocationLines
       <3>1. PICK told \in P: TRUE
         BY Linearizable
       <3>2. told.f[p] = ACK
         OBVIOUS
       <3>3. [sigma |-> told.sigma, f |-> [told.f EXCEPT ![p] = NIL]] \in P'
        BY <3>1, <3>2 DEF AugU5
      <3> QED
        BY <3>3
    <2>13. QED
      BY <2>1, <2>10, <2>11, <2>12, <2>2, <2>3, <2>4, <2>5, <2>6, <2>7, <2>8, <2>9 DEF I    
  <1>9. CASE UNCHANGED allvars
    BY <1>9, NextTypeSafety DEF Next, allvars, TypeOK, ParPointsUp, SigmaIsPartition1, SigmaIsPartition2, SigmaIsCoarse, SigmaIsFine, InvF1U1, InvF2, InvF3, InvU234, InvU5, Linearizable
  <1>10. QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9 DEF ExecStep, Next
     
THEOREM AlwaysI == Spec => []I
  <1> SUFFICES ASSUME Spec
               PROVE  []I
    OBVIOUS             
  <1> QED
    BY PTL, InitI, NextI    DEF Spec

THEOREM Linearizability == Spec => [](M # {})
  BY PTL, AlwaysI DEF I, Linearizable, M

(* Strong Linearizability *)
UniquePossibility == \A s, t \in P: s = t

LEMMA InitUniquePossibility == Init => UniquePossibility
  <1> SUFFICES ASSUME Init,
                      NEW s \in P, NEW t \in P
               PROVE  s = t
    BY DEF UniquePossibility
  <1> USE DEF Init, InitAug, UniquePossibility
  <1>1. s = [sigma |-> sigmaInit, f |-> fInit]
    OBVIOUS
  <1>2. t = [sigma |-> sigmaInit, f |-> fInit]
    OBVIOUS
  <1>3. s = t
    BY <1>1, <1>2
  <1> QED
    BY <1>3

LEMMA NextUniquePossibility == UniquePossibility /\ [Next]_allvars => UniquePossibility'
  <1> SUFFICES ASSUME UniquePossibility,
                      [Next]_allvars
               PROVE  UniquePossibility'
    OBVIOUS
  <1>1. ASSUME NEW p \in PROCSET,
               ExecF1(p)
        PROVE  UniquePossibility'
        <2> USE <1>1 DEF UniquePossibility, ExecF1, AugF1 
        <2> QED
          OBVIOUS
  <1>2. ASSUME NEW p \in PROCSET,
               ExecF2(p)
        PROVE  UniquePossibility'
    <2> SUFFICES ASSUME NEW s \in P', NEW t \in P'
                 PROVE  (s = t)'
      BY DEF UniquePossibility
        <2> USE <1>2, NextI DEF UniquePossibility, ExecF2, AugF2, TypeOK, ValidP, AtomConfigs, States, Rets, PowerSetNodes
        <2>1. CASE Par[u[p]] = u[p]
          <3> USE <2>1
          <3>1. PICK sold \in P: /\ sold.f[p] = NIL 
                                 /\ s.sigma = sold.sigma 
                                 /\ s.f = [sold.f EXCEPT ![p] = Max(sold.sigma[x[p]])]
            OBVIOUS
          <3>2. s = [sigma |-> sold.sigma, f |-> [sold.f EXCEPT ![p] = Max(sold.sigma[x[p]])]]
            BY <3>1
          <3>3. PICK told \in P: /\ told.f[p] = NIL 
                                 /\ t.sigma = told.sigma 
                                 /\ t.f = [told.f EXCEPT ![p] = Max(told.sigma[x[p]])]
            OBVIOUS
          <3>4. told = sold
            OBVIOUS
          <3>5. t = [sigma |-> sold.sigma, f |-> [sold.f EXCEPT ![p] = Max(sold.sigma[x[p]])]]
            BY <3>3, <3>4
          <3> QED
            BY <3>2, <3>5
        <2>2. CASE Par[u[p]] # u[p]
          <3> USE <2>2
          <3> QED
            OBVIOUS
        <2> QED
          BY <2>1, <2>2
  <1>3. ASSUME NEW p \in PROCSET,
               ExecF3(p)
        PROVE  UniquePossibility'
    <2> SUFFICES ASSUME NEW s \in P', NEW t \in P'
                 PROVE  (s = t)'
      BY DEF UniquePossibility
    <2> USE <1>3, NextI DEF UniquePossibility, ExecF3, AugF3, TypeOK, ValidP, AtomConfigs, States, Rets, PowerSetNodes
    <2>1. PICK sold \in P: /\ sold.f[p] = u[p]
                           /\ s.sigma = sold.sigma
                           /\ s.f = [sold.f EXCEPT ![p] = NIL]
      OBVIOUS
    <2>2. PICK told \in P: /\ told.f[p] = u[p]
                           /\ t.sigma = told.sigma
                           /\ t.f = [told.f EXCEPT ![p] = NIL]
      OBVIOUS
    <2>3. sold = told
      OBVIOUS
    <2>4. t = s
      BY <2>1, <2>2, <2>3
    <2> QED
      BY <2>4
  <1>4. ASSUME NEW p \in PROCSET,
               ExecU1(p)
        PROVE  UniquePossibility'
        <2> USE <1>4 DEF UniquePossibility, ExecU1, AugU1 
        <2> QED
          OBVIOUS
  <1>5. ASSUME NEW p \in PROCSET,
               ExecU2(p)
        PROVE  UniquePossibility'
    <2> USE <1>5, NextI DEF UniquePossibility, ExecU2, AugU2, TypeOK, ValidP, AtomConfigs, States, Rets, PowerSetNodes
    <2> SUFFICES ASSUME NEW s \in P', NEW t \in P'
                 PROVE  (s = t)'
      BY DEF UniquePossibility
    <2>1. CASE u[p] = v[p]
      <3> USE <2>1
      <3>1. PICK sold \in P: /\ sold.f[p] = NIL
                             /\ s.sigma = sold.sigma
                             /\ s.f = [sold.f EXCEPT ![p] = ACK]
        OBVIOUS
      <3>2. PICK told \in P: /\ told.f[p] = NIL
                             /\ t.sigma = told.sigma
                             /\ t.f = [told.f EXCEPT ![p] = ACK]
        OBVIOUS
      <3>3. sold = told
        OBVIOUS
      <3> QED
        BY <3>1, <3>2, <3>3
    <2>2. CASE (u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p])
      <3> USE <2>2
      <3>1. PICK sold \in P: /\ sold.f[p] = NIL
                             /\ \A z \in NodeSet: 
                                (z \in sold.sigma[x[p]] \cup sold.sigma[y[p]]) => 
                                (s.sigma[z] = sold.sigma[x[p]] \cup sold.sigma[y[p]]) 
                             /\ \A z \in NodeSet:
                                (z \notin sold.sigma[x[p]] \cup sold.sigma[y[p]]) =>
                                (s.sigma[z] = sold.sigma[z])
                             /\ s.f = [sold.f EXCEPT ![p] = ACK]
        OBVIOUS
      <3>2. PICK told \in P: /\ told.f[p] = NIL
                             /\ \A z \in NodeSet: 
                                (z \in told.sigma[x[p]] \cup told.sigma[y[p]]) => 
                                (t.sigma[z] = told.sigma[x[p]] \cup told.sigma[y[p]]) 
                             /\ \A z \in NodeSet:
                                (z \notin told.sigma[x[p]] \cup told.sigma[y[p]]) =>
                                (t.sigma[z] = told.sigma[z])
                             /\ t.f = [told.f EXCEPT ![p] = ACK]
        OBVIOUS
      <3>3. sold = told
        OBVIOUS
      <3>4. DEFINE newsigma == [z \in NodeSet |-> IF z \in sold.sigma[x[p]] \cup sold.sigma[y[p]] 
                                                    THEN sold.sigma[x[p]] \cup sold.sigma[y[p]]
                                                    ELSE sold.sigma[z]
                               ]
      <3>5. DEFINE newf     == [sold.f EXCEPT ![p] = ACK]
      <3>6. s = [sigma |-> newsigma, f |-> newf]
        BY <3>1
      <3>7. t = [sigma |-> newsigma, f |-> newf]
        BY <3>2, <3>3
      <3> QED
        BY <3>6, <3>7
    <2>3. CASE ~(u[p] = v[p]) /\ ~((u[p] < v[p] /\ Par[u[p]] = u[p]) \/ (u[p] > v[p] /\ Par[v[p]] = v[p]))
      <3> USE <2>3
      <3> QED
        OBVIOUS
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>6. ASSUME NEW p \in PROCSET,
               ExecU3(p)
        PROVE  UniquePossibility'
        <2> USE <1>6 DEF UniquePossibility, ExecU3, AugU3 
        <2> QED
          OBVIOUS
  <1>7. ASSUME NEW p \in PROCSET,
               ExecU4(p)
        PROVE  UniquePossibility'
        <2> USE <1>7 DEF UniquePossibility, ExecU4, AugU4 
        <2> QED
          OBVIOUS
  <1>8. ASSUME NEW p \in PROCSET,
               ExecU5(p)
        PROVE  UniquePossibility'
    <2> USE <1>8, NextI DEF UniquePossibility, ExecU5, AugU5, TypeOK, ValidP, AtomConfigs, States, Rets, PowerSetNodes
    <2> SUFFICES ASSUME NEW s \in P', NEW t \in P'
                 PROVE  (s = t)'
      BY DEF UniquePossibility
    <2>1. PICK sold \in P: /\ sold.f[p] = ACK
                           /\ s.sigma = sold.sigma
                           /\ s.f = [sold.f EXCEPT ![p] = NIL]
      OBVIOUS                       
    <2>2. PICK told \in P: /\ told.f[p] = ACK
                           /\ t.sigma = told.sigma
                           /\ t.f = [told.f EXCEPT ![p] = NIL]
      OBVIOUS                       
    <2>3. sold = told
      OBVIOUS
    <2> QED
      BY <2>1, <2>2, <2>3
  <1>9. CASE UNCHANGED allvars
    BY <1>9 DEF UniquePossibility, allvars
  <1>10. QED
    BY <1>1, <1>2, <1>3, <1>4, <1>5, <1>6, <1>7, <1>8, <1>9 DEF ExecStep, Next

LEMMA AlwaysUniquePossibility == Spec => []UniquePossibility
  BY PTL, InitUniquePossibility, NextUniquePossibility DEF Spec

LEMMA Cardinality1 == ASSUME Linearizable,
                             UniquePossibility
                      PROVE  Cardinality(P) = 1
<1> USE DEF Linearizable, UniquePossibility, Cardinality
<1>1. PICK t \in P: TRUE
  OBVIOUS
<1>2. P = {t}
  BY <1>1
<1>3. Cardinality(P) = 1
  BY <1>2, FS_Singleton
<1> QED
  BY <1>3

THEOREM StrongLinearizability == Spec => [](Cardinality(M) = 1)
  BY PTL, Linearizability, AlwaysUniquePossibility, Cardinality1 DEF Linearizable, M

=============================================================================
\* Modification History
\* Last modified Fri Sep 30 14:30:33 EDT 2022 by prasadjayanti
\* Last modified Fri Sep 30 14:24:57 EDT 2022 by SiddharthaJayanti
\* Created Fri Sep 30 13:41:35 EDT 2022 by SiddharthaJayanti
