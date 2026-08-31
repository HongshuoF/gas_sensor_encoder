import math
import csv
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

sensor_num = 6 # 6 in dataset example 1 and 8 in dataset example 2
time_step = 60 # time step in each inference
batch_size = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_SENSOR_COUNTS = {
    "dat_example1(classifcation_of_wine_quality).csv": 6,
    "dat_example2(classifcation_of_gas_category).csv": 8,
    "data2.csv": 4,
}

# Input: [batch_size, sensor_num * T]
# Output: outspike_list: [encoding_steps, batch_size, sensor_num]
def min_rate_encoder(self, input):

    sensor_count = int(getattr(self, "sensor_num", sensor_num))
    encoding_steps = int(getattr(self, "time_step", time_step))
    if sensor_count <= 0 or input.shape[1] % sensor_count != 0:
        raise ValueError(
            "The number of input values must be divisible by sensor_num"
        )

    outspike_list = []
    responses = input.reshape(input.shape[0], -1, sensor_count)
    min_input = responses.max(dim=1).values * encoding_steps
    input_spike = 1 / (min_input + 1)
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

def _moving_average(signal, window_size):
    """Smooth a one-dimensional signal while preserving its length."""
    window_size = max(1, min(int(window_size), signal.numel()))
    if window_size == 1:
        return signal

    left_padding = (window_size - 1) // 2
    right_padding = window_size // 2
    padded = F.pad(
        signal.reshape(1, 1, -1),
        (left_padding, right_padding),
        mode="replicate",
    )
    return F.avg_pool1d(padded, kernel_size=window_size, stride=1).flatten()


def _candidate_transition_points(acceleration, validation_window):
    """Return the most prominent local changes in response dynamics."""
    last_start = max(0, acceleration.numel() - validation_window)
    valid_acceleration = acceleration[:last_start + 1]

    if valid_acceleration.numel() <= 2:
        return torch.arange(valid_acceleration.numel(), device=acceleration.device)

    local_maxima = torch.zeros_like(valid_acceleration, dtype=torch.bool)
    local_maxima[1:-1] = (
        (valid_acceleration[1:-1] >= valid_acceleration[:-2])
        & (valid_acceleration[1:-1] >= valid_acceleration[2:])
    )

    local_indices = torch.nonzero(local_maxima, as_tuple=False).flatten()
    pool_size = min(valid_acceleration.numel(), max(12, valid_acceleration.numel() // 4))
    prominent_indices = torch.topk(valid_acceleration, k=pool_size).indices
    candidates = torch.unique(torch.cat((local_indices, prominent_indices)), sorted=True)
    return candidates


def _ordered_transition_points(
    derivative,
    acceleration,
    candidates,
    validation_window,
    theta_rise,
    theta_fall,
    theta_steady,
    minimum_window_support,
):
    """Select and validate the response and recovery boundary pairs."""
    time_points = derivative.numel()
    if time_points < 4:
        return (
            torch.zeros(4, dtype=torch.long, device=derivative.device),
            (False, False),
        )

    window_scores = []
    for candidate in candidates.tolist():
        window = derivative[candidate:min(candidate + validation_window, time_points)]
        window_scores.append(
            torch.stack(
                (
                    (window > theta_rise).sum(),
                    (window.abs() < theta_steady).sum(),
                    (window < -theta_fall).sum(),
                    (window.abs() < theta_steady).sum(),
                )
            ).to(derivative.dtype)
        )

    validation_scores = torch.stack(window_scores).transpose(0, 1)
    prominence = acceleration[candidates]
    prominence = prominence / prominence.max().clamp_min(torch.finfo(prominence.dtype).eps)
    # Derivative counts are the primary criterion; prominence breaks close ties.
    scores = validation_scores + 1e-3 * prominence.unsqueeze(0)

    minimum_gap = max(1, validation_window // 2)
    required_samples = max(
        1, math.ceil(validation_window * minimum_window_support)
    )

    def select_onset(direction_index, earliest_start):
        eligible = torch.nonzero(
            candidates >= earliest_start, as_tuple=False
        ).flatten()
        if eligible.numel() == 0:
            return 0, False
        relative_index = torch.argmax(scores[direction_index, eligible])
        candidate_index = int(eligible[relative_index])
        supported = (
            int(validation_scores[direction_index, candidate_index])
            >= required_samples
        )
        return int(candidates[candidate_index]), supported

    def select_steady(earliest_start, latest_start=None):
        eligible_mask = candidates >= earliest_start
        if latest_start is not None:
            eligible_mask &= candidates <= latest_start
        eligible = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        if eligible.numel() == 0:
            return earliest_start, False

        steady_counts = validation_scores[1, eligible]
        maximum_steady_count = steady_counts.max()
        # Candidates are sorted, so the first maximum is the steady-state onset.
        relative_index = torch.nonzero(
            steady_counts == maximum_steady_count, as_tuple=False
        ).flatten()[0]
        candidate_index = int(eligible[relative_index])
        supported = int(validation_scores[1, candidate_index]) >= required_samples
        return int(candidates[candidate_index]), supported

    t1, rise_supported = select_onset(0, 0)
    t3_start = t1 + minimum_gap if rise_supported else 0
    t3, fall_supported = select_onset(2, t3_start)

    t2_end = t3 - minimum_gap if fall_supported else None
    t2, response_steady_supported = select_steady(t1 + minimum_gap, t2_end)
    response_supported = rise_supported and response_steady_supported

    if not response_supported:
        # A false response candidate must not constrain recovery detection.
        t3, fall_supported = select_onset(2, 0)

    t4, recovery_steady_supported = select_steady(t3 + minimum_gap)
    recovery_supported = fall_supported and recovery_steady_supported

    selected = torch.tensor(
        (t1, t2, t3, t4),
        dtype=torch.long,
        device=derivative.device,
    )
    return selected, (response_supported, recovery_supported)


def _remove_outliers_and_average(
    boundaries,
    transition_support,
    time_points,
    iqr_scale,
    minimum_sensor_support,
):
    """Aggregate supported per-sensor boundaries after IQR outlier removal."""
    sensor_count = boundaries.shape[0]
    required_sensors = max(1, math.ceil(sensor_count * minimum_sensor_support))

    def aggregate(boundary_index, support_mask):
        values = boundaries[support_mask, boundary_index].float()
        first_quartile = torch.quantile(values, 0.25)
        third_quartile = torch.quantile(values, 0.75)
        iqr = third_quartile - first_quartile
        keep = (values >= first_quartile - iqr_scale * iqr) & (
            values <= third_quartile + iqr_scale * iqr
        )
        retained = values[keep]
        if retained.numel() == 0:
            retained = values
        return int(retained.mean().round().item())

    response_mask = transition_support[:, 0]
    recovery_mask = transition_support[:, 1]
    response_present = int(response_mask.sum()) >= required_sensors
    recovery_present = int(recovery_mask.sum()) >= required_sensors

    if response_present:
        t1 = max(0, min(aggregate(0, response_mask), time_points - 2))
        t2 = max(t1 + 1, min(aggregate(1, response_mask), time_points - 1))
    else:
        t1 = t2 = 0

    if recovery_present:
        earliest_t3 = t2 + 1 if response_present else 0
        if earliest_t3 > time_points - 2:
            recovery_present = False
        else:
            t3 = max(
                earliest_t3,
                min(aggregate(2, recovery_mask), time_points - 2),
            )
            t4 = max(t3 + 1, min(aggregate(3, recovery_mask), time_points - 1))

    if not recovery_present:
        # T is an exclusive end sentinel, so it is allowed to be outside the
        # valid sample-index range [0, T - 1].
        t3 = t4 = time_points

    return torch.tensor(
        (t1, t2, t3, t4), dtype=torch.long, device=boundaries.device
    )


def _detect_response_boundaries(
    responses,
    smoothing_window,
    validation_window,
    theta_rise,
    theta_fall,
    theta_steady,
    iqr_scale,
    minimum_window_support=0.5,
    minimum_sensor_support=0.5,
):
    """Detect one set of t1--t4 response boundaries for each sensor array."""
    if not 0 < minimum_window_support <= 1:
        raise ValueError("minimum_window_support must be in the interval (0, 1]")
    if not 0 < minimum_sensor_support <= 1:
        raise ValueError("minimum_sensor_support must be in the interval (0, 1]")

    batch_size, time_points, sensor_count = responses.shape
    all_boundaries = []

    with torch.no_grad():
        for batch_index in range(batch_size):
            sensor_boundaries = []
            sensor_support = []
            for sensor_index in range(sensor_count):
                signal = responses[batch_index, :, sensor_index].detach()
                centered = signal - signal[0]
                dominant_excursion = centered[centered.abs().argmax()]
                orientation = torch.where(
                    dominant_excursion < 0,
                    -torch.ones_like(dominant_excursion),
                    torch.ones_like(dominant_excursion),
                )
                oriented = centered * orientation
                normalized = oriented / oriented.abs().max().clamp_min(
                    torch.finfo(signal.dtype).eps
                )
                smoothed = _moving_average(normalized, smoothing_window)

                derivative = torch.zeros_like(smoothed)
                derivative[1:] = smoothed[1:] - smoothed[:-1]
                acceleration = torch.zeros_like(derivative)
                acceleration[1:] = (derivative[1:] - derivative[:-1]).abs()

                candidates = _candidate_transition_points(
                    acceleration, validation_window
                )
                detected_boundaries, detected_support = _ordered_transition_points(
                    derivative,
                    acceleration,
                    candidates,
                    validation_window,
                    theta_rise,
                    theta_fall,
                    theta_steady,
                    minimum_window_support,
                )
                sensor_boundaries.append(detected_boundaries)
                sensor_support.append(detected_support)

            all_boundaries.append(
                _remove_outliers_and_average(
                    torch.stack(sensor_boundaries),
                    torch.tensor(
                        sensor_support,
                        dtype=torch.bool,
                        device=responses.device,
                    ),
                    time_points,
                    iqr_scale,
                    minimum_sensor_support,
                )
            )

    return torch.stack(all_boundaries)


def increase_delta_encode(
    self,
    input,
    smoothing_window=5,
    validation_window=5,
    theta_rise=0.01,
    theta_fall=0.01,
    theta_steady=0.005,
    iqr_scale=1.5,
    minimum_window_support=0.5,
    minimum_sensor_support=0.5,
):
    """Encode responses using automatically detected response boundaries.

    ``input`` contains numeric sensor responses only and has shape
    ``[batch_size, sensor_num * T]``. Values at each time point are ordered by
    sensor. The CSV class-label column must be removed before calling this
    function. If the response pair is unsupported, ``t1 = t2 = 0`` is used;
    if the recovery pair is unsupported, ``t3 = t4 = T`` is used. 

    The code for dividing the response stage of the gas sensor is only for demonstration purposes. 
    The actual division of the sensor's response stage is carried out on the training set, and it is fixed during the actual test.
    """
    sensor_count = int(getattr(self, "sensor_num", sensor_num))
    if sensor_count <= 0 or input.shape[1] % sensor_count != 0:
        raise ValueError(
            "The number of input values must be divisible by sensor_num"
        )

    batch_size = input.shape[0]
    time_points = input.shape[1] // sensor_count

    responses = input.reshape(batch_size, time_points, sensor_count)
    validation_window = max(1, min(int(validation_window), time_points))
    boundaries = _detect_response_boundaries(
        responses,
        smoothing_window,
        validation_window,
        theta_rise,
        theta_fall,
        theta_steady,
        iqr_scale,
        minimum_window_support,
        minimum_sensor_support,
    )

    outspike_list = []
    prediction_drift = 0.01
    spike_threshold = 0.02
    prediction = responses[:, 0, :].clone()

    for time_index in range(time_points):
        current_input = responses[:, time_index, :]
        current_delta = current_input - prediction
        current_spike = (current_delta > spike_threshold).to(input.dtype)
        current_spike -= (current_delta < -spike_threshold).to(input.dtype)
        outspike_list.append(current_spike)

        in_response_transition = (
            (time_index >= boundaries[:, 0]) & (time_index < boundaries[:, 1])
        )
        in_recovery_transition = (
            (time_index >= boundaries[:, 2]) & (time_index < boundaries[:, 3])
        )
        drift = (
            in_response_transition.to(input.dtype)
            - in_recovery_transition.to(input.dtype)
        ).unsqueeze(1) * prediction_drift
        prediction = prediction + current_spike * spike_threshold + drift

    return outspike_list


def _load_csv_responses(csv_path, sensor_count):
    """Load labels and numeric response rows from one dataset CSV file."""
    labels = []
    response_rows = []

    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        for row_number, row in enumerate(csv.reader(csv_file), start=1):
            if not row:
                continue
            labels.append(row[0])
            try:
                values = [float(value) for value in row[1:]]
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric response in {csv_path.name}, row {row_number}"
                ) from error

            if not values or len(values) % sensor_count != 0:
                raise ValueError(
                    f"Row {row_number} of {csv_path.name} must contain "
                    f"sensor_num * T response values"
                )
            response_rows.append(torch.tensor(values, dtype=torch.float32))

    if not response_rows:
        raise ValueError(f"No data rows were found in {csv_path}")
    return labels, response_rows


def _encode_response_rows(response_rows, sensor_count):
    """Encode variable-length rows in batches and restore their original order."""
    rows_by_length = {}
    for row_index, row in enumerate(response_rows):
        rows_by_length.setdefault(row.numel(), []).append(row_index)

    min_rate_results = [None] * len(response_rows)
    increase_delta_results = [None] * len(response_rows)
    encoder_config = SimpleNamespace(sensor_num=sensor_count, time_step=time_step)

    for row_indices in rows_by_length.values():
        for batch_start in range(0, len(row_indices), batch_size):
            batch_row_indices = row_indices[batch_start:batch_start + batch_size]
            input_batch = torch.stack(
                [response_rows[row_index] for row_index in batch_row_indices]
            ).to(device)
            min_rate_batch = torch.stack(
                min_rate_encoder(encoder_config, input_batch), dim=0
            )
            increase_delta_batch = torch.stack(
                increase_delta_encode(encoder_config, input_batch), dim=0
            )

            for batch_index, row_index in enumerate(batch_row_indices):
                min_rate_results[row_index] = min_rate_batch[:, batch_index, :]
                increase_delta_results[row_index] = increase_delta_batch[
                    :, batch_index, :
                ]

    return min_rate_results, increase_delta_results


def main(dat_directory=None, dataset_sensor_counts=None, max_batches=1):
    """Load every CSV dataset in ``dat`` and run both spike encoders.

    Each result dictionary is keyed by CSV filename. Its value is a list in
    CSV row order; each list item is a spike tensor with shape
    ``[encoding_steps, sensor_num]``. The list representation supports datasets
    whose rows contain different numbers of time points. By default, only the
    first batch of five rows from each dataset is encoded. Set ``max_batches``
    to ``None`` to encode every row.
    """
    if dat_directory is None:
        dat_directory = Path(__file__).resolve().parent / "dat"
    else:
        dat_directory = Path(dat_directory)

    sensor_counts = DATASET_SENSOR_COUNTS.copy()
    if dataset_sensor_counts is not None:
        sensor_counts.update(dataset_sensor_counts)

    csv_paths = sorted(dat_directory.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV datasets were found in {dat_directory}")

    min_rate_encoded_results = {}
    increase_delta_encoded_results = {}

    for csv_path in csv_paths:
        if csv_path.name not in sensor_counts:
            raise ValueError(
                f"sensor_num is not configured for {csv_path.name}; pass it "
                "through dataset_sensor_counts"
            )

        sensor_count = int(sensor_counts[csv_path.name])
        _, response_rows = _load_csv_responses(csv_path, sensor_count)
        if max_batches is not None:
            response_rows = response_rows[:max_batches * batch_size]
        min_rate_rows, increase_delta_rows = _encode_response_rows(
            response_rows, sensor_count
        )
        min_rate_encoded_results[csv_path.name] = min_rate_rows
        increase_delta_encoded_results[csv_path.name] = increase_delta_rows

    return min_rate_encoded_results, increase_delta_encoded_results


if __name__ == "__main__":
    min_rate_encoded_results, increase_delta_encoded_results = main()
