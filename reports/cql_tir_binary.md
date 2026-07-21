# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=60 n_actions=10

epoch   1  loss=1.4622  td=0.4469  cql=1.0153  val_action_agreement=0.699  val_Q(data_action)=1.705  val_Q(greedy)=1.862

epoch   5  loss=1.7717  td=0.8951  cql=0.8766  val_action_agreement=0.701  val_Q(data_action)=5.197  val_Q(greedy)=5.388  dQ=+3.5260

epoch  10  loss=1.5642  td=0.6895  cql=0.8747  val_action_agreement=0.702  val_Q(data_action)=6.986  val_Q(greedy)=7.182  dQ=+1.7946

epoch  15  loss=1.5185  td=0.6497  cql=0.8688  val_action_agreement=0.698  val_Q(data_action)=7.657  val_Q(greedy)=7.826  dQ=+0.6441

epoch  20  loss=1.5008  td=0.6398  cql=0.8610  val_action_agreement=0.706  val_Q(data_action)=8.101  val_Q(greedy)=8.273  dQ=+0.4470

epoch  25  loss=1.4922  td=0.6384  cql=0.8538  val_action_agreement=0.696  val_Q(data_action)=8.030  val_Q(greedy)=8.200  dQ=-0.0730

epoch  30  loss=1.4774  td=0.6305  cql=0.8469  val_action_agreement=0.698  val_Q(data_action)=8.187  val_Q(greedy)=8.385  dQ=+0.1845

epoch  35  loss=1.4735  td=0.6302  cql=0.8434  val_action_agreement=0.702  val_Q(data_action)=8.204  val_Q(greedy)=8.396  dQ=+0.0108

epoch  40  loss=1.4656  td=0.6265  cql=0.8391  val_action_agreement=0.702  val_Q(data_action)=8.192  val_Q(greedy)=8.386  dQ=-0.0098

epoch  45  loss=1.4622  td=0.6266  cql=0.8357  val_action_agreement=0.698  val_Q(data_action)=8.319  val_Q(greedy)=8.495  dQ=+0.1094

epoch  50  loss=1.4569  td=0.6253  cql=0.8316  val_action_agreement=0.702  val_Q(data_action)=8.266  val_Q(greedy)=8.468  dQ=-0.0269

epoch  55  loss=1.4499  td=0.6218  cql=0.8281  val_action_agreement=0.701  val_Q(data_action)=8.319  val_Q(greedy)=8.507  dQ=+0.0385

epoch  60  loss=1.4461  td=0.6227  cql=0.8234  val_action_agreement=0.704  val_Q(data_action)=8.312  val_Q(greedy)=8.494  dQ=-0.0126


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.704
mean Q at clinician's action: 8.227
mean Q at learned greedy action: 8.413
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  82.1%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   5.7%
  action  3 (BOLUS_INYECTION/bin2):   3.1%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.1%
  action  7 (INFUSION/bin0):   1.3%
  action  8 (INFUSION/bin1):   3.4%
  action  9 (INFUSION/bin2):   4.3%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
