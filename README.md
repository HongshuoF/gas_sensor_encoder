# gas_sensor_spike_encoder
Here are the spike encoder for gas sensor and the example of the gas sensor response.

## Dataset Format

Example data for the spike encoder is stored in the `dat` folder in CSV format, with each row representing one sample. The first value in each row is the ground-truth class label. The remaining values are sensor responses arranged chronologically: each time point contains the responses of *n* gas sensors, where *n* is the number of sensors in the sensor array. For *T* time points, each row therefore contains *n* × *T* sensor response values, excluding the class label.

## Project Structure

```text
project/
├── dat/
│   ├── dat_example1
│   ├── dat_example2
├── work1/    
│   ├── encoder         # Accumulating rate encoding
├── work2/
│   ├── encoder         # Dual-timescale Adaptive Spike Encoder
└── README.md
```

## About
This is two different spike encoder for gas sensor response, the fuction names of both encoders are increase_delta_encode and min_rate_encoder in encoder.py. For the increase_delta_encode, the code for dividing the response stage of the gas sensor is only for demonstration purposes. The actual division of the sensor's response stage is carried out on the training set, and it is fixed during the actual test.


## How to use
Run the main fuction in encoder.py


