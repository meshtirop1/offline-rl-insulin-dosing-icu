# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=60 n_actions=10

epoch   1  loss=1.4948  td=0.4921  cql=1.0027  val_action_agreement=0.703  val_Q(data_action)=0.924  val_Q(greedy)=1.096

epoch   5  loss=1.6902  td=0.8338  cql=0.8564  val_action_agreement=0.707  val_Q(data_action)=2.981  val_Q(greedy)=3.209  dQ=+2.1137

epoch  10  loss=1.6595  td=0.8127  cql=0.8469  val_action_agreement=0.702  val_Q(data_action)=4.056  val_Q(greedy)=4.258  dQ=+1.0482

epoch  15  loss=1.6123  td=0.7733  cql=0.8390  val_action_agreement=0.699  val_Q(data_action)=4.425  val_Q(greedy)=4.609  dQ=+0.3518

epoch  20  loss=1.5945  td=0.7604  cql=0.8340  val_action_agreement=0.706  val_Q(data_action)=4.683  val_Q(greedy)=4.896  dQ=+0.2870

epoch  25  loss=1.5807  td=0.7525  cql=0.8281  val_action_agreement=0.701  val_Q(data_action)=4.700  val_Q(greedy)=4.905  dQ=+0.0082

epoch  30  loss=1.5693  td=0.7473  cql=0.8220  val_action_agreement=0.700  val_Q(data_action)=4.865  val_Q(greedy)=5.090  dQ=+0.1854

epoch  35  loss=1.5631  td=0.7441  cql=0.8190  val_action_agreement=0.701  val_Q(data_action)=4.752  val_Q(greedy)=4.965  dQ=-0.1247

epoch  40  loss=1.5550  td=0.7393  cql=0.8157  val_action_agreement=0.704  val_Q(data_action)=4.787  val_Q(greedy)=5.026  dQ=+0.0611

epoch  45  loss=1.5505  td=0.7379  cql=0.8126  val_action_agreement=0.700  val_Q(data_action)=4.842  val_Q(greedy)=5.048  dQ=+0.0211

epoch  50  loss=1.5452  td=0.7353  cql=0.8099  val_action_agreement=0.699  val_Q(data_action)=4.770  val_Q(greedy)=4.995  dQ=-0.0526

epoch  55  loss=1.5415  td=0.7344  cql=0.8071  val_action_agreement=0.699  val_Q(data_action)=4.788  val_Q(greedy)=4.997  dQ=+0.0016

epoch  60  loss=1.5380  td=0.7336  cql=0.8044  val_action_agreement=0.700  val_Q(data_action)=4.881  val_Q(greedy)=5.110  dQ=+0.1132


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.705
mean Q at clinician's action: 4.845
mean Q at learned greedy action: 5.079
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  86.8%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   2.3%
  action  3 (BOLUS_INYECTION/bin2):   2.0%
  action  4 (BOLUS_PUSH/bin0):   0.1%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.1%
  action  7 (INFUSION/bin0):   0.9%
  action  8 (INFUSION/bin1):   4.5%
  action  9 (INFUSION/bin2):   3.2%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
