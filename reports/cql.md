# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=60 n_actions=10

epoch   1  loss=1.5379  td=0.5233  cql=1.0146  val_action_agreement=0.709  val_Q(data_action)=1.616  val_Q(greedy)=1.786

epoch   5  loss=1.8761  td=0.9784  cql=0.8976  val_action_agreement=0.703  val_Q(data_action)=4.923  val_Q(greedy)=5.114  dQ=+3.3278

epoch  10  loss=1.7658  td=0.8702  cql=0.8956  val_action_agreement=0.694  val_Q(data_action)=6.638  val_Q(greedy)=6.818  dQ=+1.7041

epoch  15  loss=1.7521  td=0.8631  cql=0.8890  val_action_agreement=0.704  val_Q(data_action)=7.310  val_Q(greedy)=7.489  dQ=+0.6709

epoch  20  loss=1.7562  td=0.8723  cql=0.8839  val_action_agreement=0.689  val_Q(data_action)=7.626  val_Q(greedy)=7.795  dQ=+0.3056

epoch  25  loss=1.7510  td=0.8740  cql=0.8770  val_action_agreement=0.698  val_Q(data_action)=7.711  val_Q(greedy)=7.886  dQ=+0.0913

epoch  30  loss=1.7462  td=0.8744  cql=0.8718  val_action_agreement=0.700  val_Q(data_action)=7.861  val_Q(greedy)=8.053  dQ=+0.1674

epoch  35  loss=1.7416  td=0.8738  cql=0.8678  val_action_agreement=0.702  val_Q(data_action)=7.861  val_Q(greedy)=8.046  dQ=-0.0074

epoch  40  loss=1.7393  td=0.8747  cql=0.8646  val_action_agreement=0.699  val_Q(data_action)=7.857  val_Q(greedy)=8.050  dQ=+0.0040

epoch  45  loss=1.7329  td=0.8715  cql=0.8613  val_action_agreement=0.693  val_Q(data_action)=7.849  val_Q(greedy)=8.034  dQ=-0.0156

epoch  50  loss=1.7266  td=0.8665  cql=0.8600  val_action_agreement=0.696  val_Q(data_action)=7.886  val_Q(greedy)=8.075  dQ=+0.0407

epoch  55  loss=1.7211  td=0.8636  cql=0.8574  val_action_agreement=0.697  val_Q(data_action)=7.853  val_Q(greedy)=8.069  dQ=-0.0066

epoch  60  loss=1.7153  td=0.8605  cql=0.8548  val_action_agreement=0.696  val_Q(data_action)=7.650  val_Q(greedy)=7.827  dQ=-0.2413


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.700
mean Q at clinician's action: 7.533
mean Q at learned greedy action: 7.713
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  78.6%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   9.2%
  action  3 (BOLUS_INYECTION/bin2):   2.2%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.2%
  action  7 (INFUSION/bin0):   1.6%
  action  8 (INFUSION/bin1):   4.7%
  action  9 (INFUSION/bin2):   3.4%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
