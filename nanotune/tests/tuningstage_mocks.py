from abc import ABCMeta
import nanotune as nt
from nanotune.tuningstages.tuningstage import TuningStage


# class DummyTuningStage(TuningStage, metaclass=ABCMeta):

#     def __init__(self,
#         data_settings: Dict[str, Any],
#         setpoint_settings: Dict[str, Any],
#         readout_settings: Dict[str, Any],
#         classifier: Classifier,
#     ) -> None:

#         TuningStage.__init__(
#             self,
#             "dummy_tuning_stage",
#             data_settings,
#             setpoint_settings,
#             readout_settings,
#             classifier,
#         )


#     def take_existing_data(
#         self,
#         db_name: str = "test_dummy_data.db",
#         quality: Optional[int] = None,
#     ) -> int:
#         """"""
#         if quality is None:
#             # pick random
#             quality = np.random.randint(2, size=1)[0]

#         stage_mapping = {
#             "gatecharacterization1d": ["pinchoff"],
#             "gatecharacterization2d": ["leadcoupling"],
#             "chargediagram": ["singledot", "doubledot"],
#         }

#         labels = stage_mapping[self.stage]
#         if db_name is not None:
#             nt.set_database(db_name)
#             self.db_name = db_name

#         run_ids = []
#         for label in labels:
#             run_ids += nt.get_dataIDs(db_name, label, quality=quality)

#         rand_idx = np.random.randint(len(run_ids), size=1)[0]

#         # nt.set_database(self.db_name)

#         return run_ids[rand_idx]
