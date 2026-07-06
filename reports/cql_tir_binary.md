# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=40 n_actions=10

epoch   1  loss=1.4794  td=0.4630  cql=1.0164  val_action_agreement=0.709  val_Q(data_action)=1.724  val_Q(greedy)=1.887

epoch   5  loss=1.8110  td=0.9108  cql=0.9002  val_action_agreement=0.704  val_Q(data_action)=5.219  val_Q(greedy)=5.402  dQ=+3.5155

epoch  10  loss=1.6592  td=0.7598  cql=0.8994  val_action_agreement=0.695  val_Q(data_action)=7.028  val_Q(greedy)=7.200  dQ=+1.7979

epoch  15  loss=1.6471  td=0.7535  cql=0.8936  val_action_agreement=0.705  val_Q(data_action)=7.713  val_Q(greedy)=7.878  dQ=+0.6778

epoch  20  loss=1.6507  td=0.7637  cql=0.8870  val_action_agreement=0.693  val_Q(data_action)=8.050  val_Q(greedy)=8.212  dQ=+0.3336

epoch  25  loss=1.6492  td=0.7696  cql=0.8796  val_action_agreement=0.699  val_Q(data_action)=8.088  val_Q(greedy)=8.254  dQ=+0.0423

epoch  30  loss=1.6439  td=0.7690  cql=0.8749  val_action_agreement=0.703  val_Q(data_action)=8.324  val_Q(greedy)=8.507  dQ=+0.2533

epoch  35  loss=1.6425  td=0.7705  cql=0.8721  val_action_agreement=0.703  val_Q(data_action)=8.329  val_Q(greedy)=8.506  dQ=-0.0017

epoch  40  loss=1.6405  td=0.7715  cql=0.8690  val_action_agreement=0.701  val_Q(data_action)=8.288  val_Q(greedy)=8.469  dQ=-0.0368


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.705
mean Q at clinician's action: 8.188
mean Q at learned greedy action: 8.371
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  78.9%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   6.4%
  action  3 (BOLUS_INYECTION/bin2):   3.4%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.4%
  action  7 (INFUSION/bin0):   1.3%
  action  8 (INFUSION/bin1):   5.2%
  action  9 (INFUSION/bin2):   4.5%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
