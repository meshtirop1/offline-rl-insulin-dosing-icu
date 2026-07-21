# Discrete CQL training

gamma=0.95 alpha_cql=1.0 lr=0.001 batch_size=512 epochs=60 n_actions=10

epoch   1  loss=1.5151  td=0.5027  cql=1.0124  val_action_agreement=0.702  val_Q(data_action)=1.590  val_Q(greedy)=1.750

epoch   5  loss=1.8264  td=0.9530  cql=0.8734  val_action_agreement=0.704  val_Q(data_action)=4.916  val_Q(greedy)=5.113  dQ=+3.3626

epoch  10  loss=1.6458  td=0.7772  cql=0.8685  val_action_agreement=0.703  val_Q(data_action)=6.572  val_Q(greedy)=6.764  dQ=+1.6512

epoch  15  loss=1.6020  td=0.7408  cql=0.8611  val_action_agreement=0.699  val_Q(data_action)=7.219  val_Q(greedy)=7.394  dQ=+0.6304

epoch  20  loss=1.5858  td=0.7296  cql=0.8562  val_action_agreement=0.704  val_Q(data_action)=7.617  val_Q(greedy)=7.796  dQ=+0.4015

epoch  25  loss=1.5753  td=0.7253  cql=0.8500  val_action_agreement=0.697  val_Q(data_action)=7.620  val_Q(greedy)=7.799  dQ=+0.0028

epoch  30  loss=1.5629  td=0.7196  cql=0.8433  val_action_agreement=0.698  val_Q(data_action)=7.763  val_Q(greedy)=7.972  dQ=+0.1735

epoch  35  loss=1.5600  td=0.7195  cql=0.8405  val_action_agreement=0.701  val_Q(data_action)=7.744  val_Q(greedy)=7.938  dQ=-0.0340

epoch  40  loss=1.5546  td=0.7180  cql=0.8366  val_action_agreement=0.702  val_Q(data_action)=7.766  val_Q(greedy)=7.979  dQ=+0.0404

epoch  45  loss=1.5485  td=0.7157  cql=0.8328  val_action_agreement=0.697  val_Q(data_action)=7.875  val_Q(greedy)=8.057  dQ=+0.0786

epoch  50  loss=1.5430  td=0.7143  cql=0.8288  val_action_agreement=0.701  val_Q(data_action)=7.805  val_Q(greedy)=8.015  dQ=-0.0419

epoch  55  loss=1.5383  td=0.7128  cql=0.8255  val_action_agreement=0.699  val_Q(data_action)=7.867  val_Q(greedy)=8.056  dQ=+0.0411

epoch  60  loss=1.5316  td=0.7101  cql=0.8214  val_action_agreement=0.702  val_Q(data_action)=7.843  val_Q(greedy)=8.045  dQ=-0.0114


## Test-set results
action agreement with clinician (greedy policy == logged action): 0.705
mean Q at clinician's action: 7.766
mean Q at learned greedy action: 7.970
(higher Q at greedy than at data action is expected -- that's the policy improvement signal; the open question, deferred to formal OPE, is whether that Q gap reflects a real improvement or overestimation bias)

## Learned greedy policy's action distribution on test states
  action  0 (NONE/bin-1):  86.1%
  action  1 (BOLUS_INYECTION/bin0):   0.0%
  action  2 (BOLUS_INYECTION/bin1):   3.6%
  action  3 (BOLUS_INYECTION/bin2):   1.8%
  action  4 (BOLUS_PUSH/bin0):   0.0%
  action  5 (BOLUS_PUSH/bin1):   0.0%
  action  6 (BOLUS_PUSH/bin2):   0.1%
  action  7 (INFUSION/bin0):   1.6%
  action  8 (INFUSION/bin1):   3.3%
  action  9 (INFUSION/bin2):   3.5%

## Caveat
This is training-time diagnostics only (TD loss convergence, CQL conservatism, action agreement). It is NOT a validated estimate of clinical benefit -- that requires formal off-policy evaluation (FQE / weighted importance sampling / doubly robust) against a real outcome (hypoglycemia incidence, mortality), which needs the PhysioNet ADMISSIONS/PATIENTS outcome linkage described in the project README. Do not treat 'Q(greedy) > Q(data)' above as evidence the learned policy is safer.
