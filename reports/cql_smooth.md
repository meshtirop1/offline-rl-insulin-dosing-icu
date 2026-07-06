# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=40 n_actions=10

epoch   1  loss=1.5255  td=0.5183  cql=1.0071  val_action_agreement=0.707  val_Q(data_action)=0.987  val_Q(greedy)=1.178

epoch   5  loss=1.8048  td=0.9217  cql=0.8830  val_action_agreement=0.701  val_Q(data_action)=3.046  val_Q(greedy)=3.265  dQ=+2.0869

epoch  10  loss=1.8083  td=0.9344  cql=0.8739  val_action_agreement=0.697  val_Q(data_action)=4.024  val_Q(greedy)=4.223  dQ=+0.9579

epoch  15  loss=1.8003  td=0.9337  cql=0.8666  val_action_agreement=0.705  val_Q(data_action)=4.511  val_Q(greedy)=4.714  dQ=+0.4913

epoch  20  loss=1.7969  td=0.9361  cql=0.8608  val_action_agreement=0.698  val_Q(data_action)=4.743  val_Q(greedy)=4.937  dQ=+0.2234

epoch  25  loss=1.7958  td=0.9409  cql=0.8550  val_action_agreement=0.702  val_Q(data_action)=4.797  val_Q(greedy)=4.989  dQ=+0.0522

epoch  30  loss=1.7913  td=0.9412  cql=0.8501  val_action_agreement=0.704  val_Q(data_action)=4.761  val_Q(greedy)=4.977  dQ=-0.0120

epoch  35  loss=1.7865  td=0.9396  cql=0.8469  val_action_agreement=0.703  val_Q(data_action)=4.830  val_Q(greedy)=5.033  dQ=+0.0558

epoch  40  loss=1.7843  td=0.9400  cql=0.8443  val_action_agreement=0.704  val_Q(data_action)=4.841  val_Q(greedy)=5.049  dQ=+0.0153


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.708
mean Q at clinician's action: 4.756
mean Q at learned greedy action: 4.965
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  84.6%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   3.1%
  action  3 (BOLUS_INYECTION/bin2):   2.7%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.4%
  action  7 (INFUSION/bin0):   1.3%
  action  8 (INFUSION/bin1):   4.3%
  action  9 (INFUSION/bin2):   3.5%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
