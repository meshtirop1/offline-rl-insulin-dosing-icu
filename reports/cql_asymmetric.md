# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=60 n_actions=10

epoch   1  loss=1.5246  td=0.5121  cql=1.0125  val_action_agreement=0.701  val_Q(data_action)=1.577  val_Q(greedy)=1.738

epoch   5  loss=1.8270  td=0.9541  cql=0.8729  val_action_agreement=0.704  val_Q(data_action)=4.855  val_Q(greedy)=5.050  dQ=+3.3112

epoch  10  loss=1.6492  td=0.7797  cql=0.8695  val_action_agreement=0.703  val_Q(data_action)=6.517  val_Q(greedy)=6.714  dQ=+1.6641

epoch  15  loss=1.6081  td=0.7450  cql=0.8632  val_action_agreement=0.698  val_Q(data_action)=7.176  val_Q(greedy)=7.350  dQ=+0.6359

epoch  20  loss=1.5899  td=0.7339  cql=0.8560  val_action_agreement=0.705  val_Q(data_action)=7.564  val_Q(greedy)=7.744  dQ=+0.3940

epoch  25  loss=1.5797  td=0.7310  cql=0.8487  val_action_agreement=0.698  val_Q(data_action)=7.589  val_Q(greedy)=7.769  dQ=+0.0255

epoch  30  loss=1.5694  td=0.7271  cql=0.8422  val_action_agreement=0.700  val_Q(data_action)=7.691  val_Q(greedy)=7.895  dQ=+0.1256

epoch  35  loss=1.5656  td=0.7265  cql=0.8391  val_action_agreement=0.699  val_Q(data_action)=7.699  val_Q(greedy)=7.895  dQ=+0.0007

epoch  40  loss=1.5599  td=0.7245  cql=0.8354  val_action_agreement=0.704  val_Q(data_action)=7.725  val_Q(greedy)=7.935  dQ=+0.0397

epoch  45  loss=1.5538  td=0.7218  cql=0.8320  val_action_agreement=0.696  val_Q(data_action)=7.824  val_Q(greedy)=8.002  dQ=+0.0672

epoch  50  loss=1.5482  td=0.7200  cql=0.8282  val_action_agreement=0.700  val_Q(data_action)=7.742  val_Q(greedy)=7.955  dQ=-0.0476

epoch  55  loss=1.5434  td=0.7178  cql=0.8256  val_action_agreement=0.700  val_Q(data_action)=7.805  val_Q(greedy)=7.994  dQ=+0.0395

epoch  60  loss=1.5379  td=0.7162  cql=0.8217  val_action_agreement=0.701  val_Q(data_action)=7.829  val_Q(greedy)=8.024  dQ=+0.0298


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.703
mean Q at clinician's action: 7.764
mean Q at learned greedy action: 7.965
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  84.8%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   4.1%
  action  3 (BOLUS_INYECTION/bin2):   2.2%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.2%
  action  7 (INFUSION/bin0):   1.2%
  action  8 (INFUSION/bin1):   4.2%
  action  9 (INFUSION/bin2):   3.3%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
