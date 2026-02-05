import json
import os

class MappingManager:
    def __init__(self, mapping_file_path):
        self.mapping_file_path = mapping_file_path
        self.mapping = self._load_mapping()

    def _load_mapping(self):
        if not os.path.exists(self.mapping_file_path):
            raise FileNotFoundError(f"Mapping file not found: {self.mapping_file_path}")
        with open(self.mapping_file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_info_for_sheet(self, sheet_name):
        """Returns {'lang': ..., 'code': ...} for a given sheet name."""
        return self.mapping.get(sheet_name)

    def get_all_target_sheets(self, available_sheets):
        """Returns a list of sheet names that exist in the workbook and have a mapping, excluding source."""
        targets = []
        for sheet in available_sheets:
            if sheet == self.source_sheet_name:
                continue
            if sheet in self.mapping:
                targets.append(sheet)
        return targets

    def get_source_info(self):
        return self.mapping.get(self.source_sheet_name)

    def is_source_sheet(self, sheet_name):
        return sheet_name == self.source_sheet_name
