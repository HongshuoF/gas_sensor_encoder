"""Build per-sensor probability-distribution mapping polynomials.

CSV rows have the following layout::

    label, sensor_0_t0, ..., sensor_n_t0, sensor_0_t1, ..., sensor_n_t1, ...

The functions below select one time group, normalize every sensor over all
samples, count its values in equal-width bins, and interpolate the bin
frequencies with one polynomial per sensor.
"""

import csv
from pathlib import Path

import torch


DEFAULT_BIN_COUNT = 20


def _load_csv_responses(csv_path, sensor_count, x=0):
    """Load the labels and sensor responses at time-group ``x``.
    Args:
        csv_path: Path to one CSV dataset in ``dat``.
        sensor_count: Number of sensors at each time point.
        x: time-group index that is sampled in steady stage. The first time group is ``x=0``.

    Returns:
        A pair ``(labels, responses)``. ``labels`` is a list of strings and
        ``responses`` is a float tensor shaped ``[sample_count, sensor_count]``.
    """
    csv_path = Path(csv_path)

    labels = []
    selected_responses = []
    selected_start = x * sensor_count
    selected_end = selected_start + sensor_count

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
            if x >= time_group_count:
                raise IndexError(
                    f"x={x} is outside row {row_number} of {csv_path.name}; "
                )

            try:
                selected_values = [
                    float(value)
                    for value in numeric_fields[selected_start:selected_end]
                ]
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric response in {csv_path.name}, row {row_number}, "
                    f"time group {x}"
                ) from error

            labels.append(row[0])
            selected_responses.append(selected_values)

    responses = torch.tensor(selected_responses, dtype=torch.float32)
    return labels, responses


def data_normalization(responses):
    """Min-max normalize each sensor column independently to ``[0, 1]``.
    """
    if responses.ndim != 2 or responses.numel() == 0:
        raise ValueError("responses must have shape [sample_count, sensor_count]")
    sensor_minimums = responses.amin(dim=0, keepdim=True)
    sensor_ranges = responses.amax(dim=0, keepdim=True) - sensor_minimums
    non_constant = sensor_ranges > 0
    safe_ranges = torch.where(
        non_constant, sensor_ranges, torch.ones_like(sensor_ranges)
    )
    response_norm = (responses - sensor_minimums) / safe_ranges
    response_norm = torch.where(
        non_constant, response_norm, torch.zeros_like(response_norm)
    )
    return response_norm.clamp(0.0, 1.0)


def data_frequency_count(response_norm, bin_count=DEFAULT_BIN_COUNT):
    """Return each sensor's frequency distribution over equal-width bins.

    The default 20 bins are ``[0, 0.05)``, ..., ``[0.95, 1]``. The result has shape ``[sensor_count, bin_count]`` and
    every sensor row sums to one.
    """

    if response_norm.ndim != 2 or response_norm.numel() == 0:
        raise ValueError(
            "response_norm must have shape [sample_count, sensor_count]"
        )
    normalized = response_norm
    normalized = normalized.clamp(0.0, 1.0)

    bin_indices = torch.floor(normalized * bin_count).to(torch.long)
    bin_indices.clamp_(max=bin_count - 1)

    sample_count, sensor_count = normalized.shape
    freq_distribute = torch.zeros(
        (sensor_count, bin_count),
        dtype=torch.float64,
        device=normalized.device,
    )
    for sensor_index in range(sensor_count):
        freq_distribute[sensor_index] = torch.bincount(
            bin_indices[:, sensor_index], minlength=bin_count
        ).to(torch.float64)
    return freq_distribute / sample_count


def _multiply_by_linear_factor(coefficients, root):
    """Multiply descending power-basis coefficients by ``(x - root)``."""
    product = torch.zeros(
        coefficients.numel() + 1,
        dtype=coefficients.dtype,
        device=coefficients.device,
    )
    product[:-1] += coefficients
    product[1:] -= root * coefficients
    return product


def data_interpolation(freq_distribute):
    """Lagrange-interpolate each sensor's frequency distribution.

    A bin's interpolation coordinate is its midpoint. With 20 bins these are
    ``0.025, 0.075, ..., 0.975``. The returned tensor has shape
    ``[sensor_count, bin_count]``. Each row contains one sensor polynomial's
    coefficients in descending-power order.
    """
    if not isinstance(freq_distribute, torch.Tensor):
        frequencies = torch.as_tensor(freq_distribute, dtype=torch.float64)
    else:
        frequencies = freq_distribute.to(torch.float64)
    if frequencies.ndim != 2:
        raise ValueError(
            "freq_distribute must have shape [sensor_count, bin_count]"
        )
    sensor_count, bin_count = frequencies.shape

    midpoints = (
        torch.arange(
            bin_count, dtype=torch.float64, device=frequencies.device
        )
        + 0.5
    ) / bin_count
    coefficients = torch.zeros_like(frequencies)

    for point_index in range(bin_count):
        basis = torch.ones(
            1, dtype=torch.float64, device=frequencies.device
        )
        denominator = torch.ones(
            (), dtype=torch.float64, device=frequencies.device
        )
        point = midpoints[point_index]
        for other_index in range(bin_count):
            if other_index == point_index:
                continue
            other_point = midpoints[other_index]
            basis = _multiply_by_linear_factor(basis, other_point)
            denominator *= point - other_point

        coefficients += (
            frequencies[:, point_index] / denominator
        ).unsqueeze(1) * basis.unsqueeze(0)

    return coefficients


def mapping_fuction_calculate(coefficients):
    """Calculate a monotonic F(x) at all bin edges."""
    coefficients = coefficients.to(torch.float64)
    bin_count = coefficients.shape[1]
    midpoints = (
        torch.arange(bin_count, dtype=torch.float64, device=coefficients.device)
        + 0.5
    ) / bin_count

    frequency = torch.zeros_like(coefficients)
    for coefficient in coefficients.unbind(dim=1):
        frequency = frequency * midpoints + coefficient.unsqueeze(1)
    frequency.clamp_min_(0.0)

    mapping_function = torch.cat(
        (torch.zeros_like(frequency[:, :1]), frequency.cumsum(dim=1)),
        dim=1,
    )
    total_area = mapping_function[:, -1:].clamp_min(
        torch.finfo(mapping_function.dtype).eps
    )
    return mapping_function / total_area


def data_mapping(response_norm, mapping_function):
    """Map normalized sensor data through F(x) using linear interpolation."""
    values = response_norm.clamp(0.0, 1.0).transpose(0, 1)
    position = values * (mapping_function.shape[1] - 1)
    left_index = position.floor().to(torch.long)
    right_index = (left_index + 1).clamp(max=mapping_function.shape[1] - 1)
    weight = position - left_index

    left_value = torch.gather(mapping_function, 1, left_index)
    right_value = torch.gather(mapping_function, 1, right_index)
    mapped_data = left_value + weight * (right_value - left_value)
    return mapped_data.transpose(0, 1).to(response_norm.dtype)


def build_mapping(csv_path, sensor_count, x, bin_count=DEFAULT_BIN_COUNT):
    """Run the complete mapping pipeline for one dataset and time group."""
    labels, responses = _load_csv_responses(csv_path, sensor_count, x)
    normalized = data_normalization(responses)
    frequencies = data_frequency_count(normalized, bin_count)
    coefficients = data_interpolation(frequencies)
    return labels, normalized, coefficients


def main():
    sensor_count = 4  # sensor number in dataset: 6 in example1 and 4 in example2
    x = 25         # sample point in dataset: 25 in example1 and 25  in example2
    csv_path = (
        Path(__file__).resolve().parents[1]
        / "dat"
        # / "dat_example1(classifcation_of_wine_quality).csv"
        / "dat_example2(classifcation_of_gas_category).csv"
    )

    labels, normalized, coefficients = build_mapping(
        csv_path,
        sensor_count=sensor_count,
        x=x,
    )
    mapping_function = mapping_fuction_calculate(coefficients)
    mapped_data = data_mapping(normalized, mapping_function)

    output_path = csv_path.with_name(f"{csv_path.stem}_mapped.csv")
    with output_path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.writer(csv_file)
        for label, values in zip(labels, mapped_data.detach().cpu().tolist()):
            writer.writerow([label, *(f"{value:.6f}" for value in values)])

    return mapped_data


if __name__ == "__main__":
    main()
