declare module 'events' {
  /** Minimal EventEmitter interface used by the SDK without relying on @types/node. */
  export class EventEmitter {
    on(event: string | symbol, listener: (...args: any[]) => void): this;
    emit(event: string | symbol, ...args: any[]): boolean;
  }
}
