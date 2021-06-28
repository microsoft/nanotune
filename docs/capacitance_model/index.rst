
Capacitance Model
=================


The capacitance model:
It simulated quantum dots as a system of capacitors connecting charge and voltages nodes, i.e. dots and gates respectively. Beside calculating integer number of charges on charge nodes, the model doesn't involve any dot or material physics.

Synthetic data
--------------
Training data used for the "Evaluation of synthetic and experimental training data …" (https://arxiv.org/abs/2005.08131) comes form two sources: the Qflow-lite dataset and data generated using the capacitance model. Beside transport data, the Qflow-lite dataset also provides charge sensing data (second plot of each measurement).
The general problem with synthetic data or simple models such as the capacitance model is that their ability to reproduce real device behavior is limited. In the examples below for example, only two out of many possible double dot states are covered. The situation is better for single dots as they are a lot simpler.
Nanotune's implementation allows to sweep arbitrary gates of an N-dot system. It implements gate cross-talk, which manifests itself in the shift of transport features in gate voltage if a nearby gate is changed. The shortcoming of these models is that they only represent well-defined dot and don’t reproduce 'poor' regimes as shown above. They don't allow to test tuning sequences aiming to tune between no-dot and well-defined regimes.