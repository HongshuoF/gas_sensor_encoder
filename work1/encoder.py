
import csv
import torch
from pathlib import Path

def _load_csv_responses(csv_path, sensor_count):
    """Load the labels and sensor responses.
    Args:
        csv_path: Path to one CSV dataset in ``dat``.
        sensor_count: Number of sensors at each time point.

    Returns:
        A pair ``(labels, responses)``. ``labels`` is a list of strings and
        ``responses`` is a float tensor shaped ``[sample_count, sensor_count]``.
    """
    csv_path = Path(csv_path)

    labels = []
    selected_responses = []
    selected_start = 0
    selected_end = sensor_count

    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row or all(not field.strip() for field in row):
                continue

            numeric_fields = row[1:]
            if len(numeric_fields) % sensor_count != 0:
                raise ValueError(
                    f"Row {row_number} of {csv_path.name} must contain "
                )
            time_group_count = len(numeric_fields) // sensor_count

            try:
                selected_values = [
                    float(value)
                    for value in numeric_fields[selected_start:selected_end]
                ]
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric response in {csv_path.name}, row {row_number}, "
                ) from error

            labels.append(row[0])
            selected_responses.append(selected_values)

    responses = torch.tensor(selected_responses, dtype=torch.float32)
    return labels, responses

def accumulating_rate_encoder(input, sensor_count, encoding_steps):
    """
    Encodes the input into a series of spikes using an accumulating rate coding scheme.
    Args:
        input: A float tensor shaped ``[sample_count, sensor_count]`` representing the mapped sensor responses.
        sensor_count: Number of sensors at each time point.
        encoding_steps: Number of time steps for encoding.

    Returns:
        A list of float tensors, each shaped ``[sample_count, sensor_count]``, representing the encoded spikes.
    """
    if sensor_count <= 0 or input.shape[1] % sensor_count != 0:
        raise ValueError(
            "The number of input values must be divisible by sensor_count"
        )

    outspike_list = []
    responses = input.reshape(input.shape[0], -1, sensor_count)
    input_spike = responses
    # min_input = responses.max(dim=1).values * encoding_steps
    # input_spike = 1 / (min_input + 1)
    input_accu = input_spike.clone()
    for _ in range(encoding_steps):
        # Generate rate-coded spikes using an accumulator.
        accu_temp = input_accu.clone() - 1
        accu_temp = accu_temp.floor()
        accu_temp = accu_temp.bool()
        accu_temp = ~accu_temp
        accu_temp = accu_temp.float()
        outspike_list.append(accu_temp)
        input_accu = input_accu + input_spike - accu_temp
    return outspike_list


def main():
    sensor_count = 6  # sensor number in dataset: 6 in example1 and 4 in example2
    time_step = 10

    csv_path = (
        Path(__file__).resolve().parents[1]
        / "dat"
        / "dat_example1(classifcation_of_wine_quality)_mapped.csv"
    )
    labels, responses = _load_csv_responses(csv_path, sensor_count)
    outspike_list = accumulating_rate_encoder(responses, sensor_count, time_step)
    return outspike_list

if __name__ == "__main__":
    main()