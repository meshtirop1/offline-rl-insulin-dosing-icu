# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=40 n_actions=10

epoch   1  loss=1.5473  td=0.5327  cql=1.0146  val_action_agreement=0.709  val_Q(data_action)=1.610  val_Q(greedy)=1.780

epoch   5  loss=1.8853  td=0.9879  cql=0.8974  val_action_agreement=0.703  val_Q(data_action)=4.886  val_Q(greedy)=5.079  dQ=+3.2985

epoch  10  loss=1.7666  td=0.8733  cql=0.8933  val_action_agreement=0.694  val_Q(data_action)=6.606  val_Q(greedy)=6.786  dQ=+1.7071

epoch  15  loss=1.7511  td=0.8631  cql=0.8879  val_action_agreement=0.703  val_Q(data_action)=7.278  val_Q(greedy)=7.455  dQ=+0.6685

epoch  20  loss=1.7550  td=0.8723  cql=0.8826  val_action_agreement=0.692  val_Q(data_action)=7.563  val_Q(greedy)=7.734  dQ=+0.2796

epoch  25  loss=1.7510  td=0.8747  cql=0.8763  val_action_agreement=0.700  val_Q(data_action)=7.632  val_Q(greedy)=7.809  dQ=+0.0747

epoch  30  loss=1.7459  td=0.8745  cql=0.8713  val_action_agreement=0.701  val_Q(data_action)=7.802  val_Q(greedy)=7.990  dQ=+0.1807

epoch  35  loss=1.7410  td=0.8735  cql=0.8675  val_action_agreement=0.701  val_Q(data_action)=7.825  val_Q(greedy)=8.007  dQ=+0.0177

epoch  40  loss=1.7413  td=0.8764  cql=0.8648  val_action_agreement=0.700  val_Q(data_action)=7.815  val_Q(greedy)=8.009  dQ=+0.0021


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.703
mean Q at clinician's action: 7.709
mean Q at learned greedy action: 7.907
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  82.8%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   3.4%
  action  3 (BOLUS_INYECTION/bin2):   2.9%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.5%
  action  7 (INFUSION/bin0):   2.1%
  action  8 (INFUSION/bin1):   2.9%
  action  9 (INFUSION/bin2):   5.4%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
