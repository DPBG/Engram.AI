/**
 * SDK core types and plugin registration helpers for Humanoid Active Learning AI.
 *
 * These definitions form the open-source surface area for registering sensors and actuators
 * and exchanging observations and actions with the moral kernel.
 */

import { EventEmitter } from 'events';

/** Event bus used internally by the SDK for observations and actions. */
export const sdkBus = new EventEmitter();

/** Unique identifier for correlating observations and actions. */
export type TraceId = string;

/**
 * Observation emitted by a sensor plugin.
 */
export interface Observation<T = unknown> {
  /** traceId correlating this observation with resulting actions */
  traceId: TraceId;
  /** identifier describing where the data originated */
  provenance: string;
  /** sensor payload */
  data: T;
  /** optional timestamp in milliseconds since epoch */
  timestamp?: number;
}

/**
 * Action proposed by a planner or other module.
 */
export interface ActionProposal<T = unknown> {
  traceId: TraceId;
  provenance: string;
  /** command or structured payload describing the action */
  action: T;
}

/** Decision types returned by the moral kernel for a proposed action. */
export enum KernelDecisionType {
  ALLOW = 'ALLOW',
  TRANSFORM = 'TRANSFORM',
  DENY = 'DENY',
}

/**
 * Result of kernel evaluation. When `type` is `TRANSFORM`, optional
 * transformation options may be provided describing safe alterations of the
 * original action.
 */
export interface KernelDecision<T = unknown> {
  /** overall decision category */
  type: KernelDecisionType;
  /** possible transformations suggested by the kernel */
  transformations?: ActionProposal<T>[];
}

/**
 * Outcome of a proposed action after evaluation by the kernel.
 */
export interface Outcome<T = unknown> {
  traceId: TraceId;
  decision: KernelDecision<T>;
  /** original proposal before any transformation */
  original: ActionProposal<T>;
  /** final action after applying a selected transformation */
  final?: ActionProposal<T>;
  /** optional human-readable reason for the decision */
  reason?: string;
}

/**
 * Definition of an actuator safety envelope used to constrain executions.
 */
export interface SafetyEnvelope<T = unknown> {
  /**
   * Validate that the provided outcome falls within the acceptable envelope.
   * If not supplied, all outcomes are assumed safe.
   */
  validate?: (outcome: Outcome<T>) => boolean;
  /** optional human-readable description */
  description?: string;
}

/** Basic logger interface. */
export interface Logger {
  debug: (...args: unknown[]) => void;
  info: (...args: unknown[]) => void;
  warn: (...args: unknown[]) => void;
  error: (...args: unknown[]) => void;
}

/** Default console-based logger. */
export const logger: Logger = {
  debug: (...args) => console.debug('[debug]', ...args),
  info: (...args) => console.info('[info]', ...args),
  warn: (...args) => console.warn('[warn]', ...args),
  error: (...args) => console.error('[error]', ...args),
};

/** Map of registered sensors keyed by id. */
const sensors = new Map<string, SensorPlugin>();

/** Interface for sensor plugins. */
export interface SensorPlugin {
  /** unique identifier for this sensor */
  id: string;
  /** start the sensor; emit observations via the provided callback */
  start: (emit: (obs: Observation) => void, log: Logger) => void;
  /** optional stop hook */
  stop?: () => void;
}

/** Interface for actuator plugins. */
export interface ActuatorPlugin {
  /** unique identifier for this actuator */
  id: string;
  /** safety envelope constraining actions for this actuator */
  envelope?: SafetyEnvelope;
  /** execute an action outcome */
  execute: (outcome: Outcome, log: Logger) => Promise<void> | void;
}

/**
 * Register a sensor plugin. Sensors are inert until `startSensor` is called.
 */
export function registerSensor(plugin: SensorPlugin): void {
  sensors.set(plugin.id, plugin);
  logger.info(`Sensor registered: ${plugin.id}`);
}

/** Start a previously registered sensor. */
export function startSensor(id: string): void {
  const plugin = sensors.get(id);
  if (!plugin) {
    logger.warn(`Sensor ${id} not registered`);
    return;
  }
  plugin.start((obs) => {
    sdkBus.emit('observation', obs);
    logger.debug('Observation emitted', obs);
  }, logger);
}

/** Stop a running sensor if it provides a stop hook. */
export function stopSensor(id: string): void {
  const plugin = sensors.get(id);
  if (!plugin) {
    logger.warn(`Sensor ${id} not registered`);
    return;
  }
  plugin.stop?.();
}

/**
 * Register an actuator plugin which listens for action outcomes.
 */
export function registerActuator(plugin: ActuatorPlugin): void {
  sdkBus.on('action', (outcome: Outcome) => {
    if (outcome.decision.type === KernelDecisionType.DENY) {
      logger.warn(`Action denied for traceId ${outcome.traceId}`);
      return;
    }
    if (plugin.envelope && plugin.envelope.validate && !plugin.envelope.validate(outcome)) {
      logger.warn(`Outcome for traceId ${outcome.traceId} outside safety envelope of actuator ${plugin.id}`);
      return;
    }
    plugin.execute(outcome, logger);
  });
}

/**
 * Emit an action outcome onto the SDK bus. This would typically be called
 * after the moral kernel evaluates an action proposal.
 */
export function emitOutcome(outcome: Outcome): void {
  sdkBus.emit('action', outcome);
  logger.debug('Action outcome emitted', outcome);
}
