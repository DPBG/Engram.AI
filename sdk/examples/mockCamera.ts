/**
 * Example sensor plugin that simulates a camera and emits mock frames.
 */
import { registerSensor, startSensor, stopSensor, SensorPlugin, Observation } from '../index';

let interval: ReturnType<typeof setInterval> | undefined;

const mockCamera: SensorPlugin = {
  id: 'mock-camera',
  start: (emit, log) => {
    log.info('Mock camera started');
    interval = setInterval(() => {
      const frame: Observation<{ image: string }> = {
        traceId: Math.random().toString(36).slice(2),
        provenance: 'mock-camera',
        data: { image: '<pretend binary data>' },
        timestamp: Date.now(),
      };
      emit(frame);
    }, 1000);
  },
  stop: () => {
    if (interval) {
      clearInterval(interval);
    }
  },
};

// Register and start the mock camera sensor with the SDK.
registerSensor(mockCamera);
startSensor('mock-camera');

// Stop the sensor after 5 seconds for demonstration purposes.
setTimeout(() => stopSensor('mock-camera'), 5000);
