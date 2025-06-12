from pathlib import Path

current_file_directory = Path(__file__).resolve().parent
test_data_path = Path(f"{current_file_directory}/reference_data")


def test_protein_group_reader(): ...
