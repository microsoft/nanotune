from typing import List, Optional
import nanotune as nt

ALLOWED_CATEGORIES = list(dict(nt.config["core"]["features"]).keys())


class MockClassifer:
    def __init__(self, category):
        assert category in ALLOWED_CATEGORIES
        self.category = category

    def predict(
        self,
        dataid: int,
        db_name: str,
        db_folder: Optional[str] = None,
    ) -> List[int]:
        # if self.category == 'singledot':
        #     return [0]
        # else:
        return [1]
