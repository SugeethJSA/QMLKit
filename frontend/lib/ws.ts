"use client";

import { useEffect, useRef, useState } from "react";
import { wsUrl } from "./api";

/** Reconnecting WebSocket hook with exponential backoff. */
export function useTelemetrySocket<T>(path: string, onMessage: (data: T) => void) {
  const [connected, setConnected] = useState(false);
  const handlerRef = useRef(onMessage);

  useEffect(() => {
    handlerRef.current = onMessage;
  }, [onMessage]);

  useEffect(() => {
    let socket: WebSocket | null = null;
    let closed = false;
    let attempt = 0;
    let timer: ReturnType<typeof setTimeout>;

    const connect = () => {
      socket = new WebSocket(wsUrl(path));
      socket.onopen = () => {
        setConnected(true);
        attempt = 0;
      };
      socket.onmessage = (event) => {
        try {
          handlerRef.current(JSON.parse(event.data));
        } catch {
          /* ignore malformed frames */
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!closed) {
          const delay = Math.min(8000, 500 * 2 ** attempt++);
          timer = setTimeout(connect, delay);
        }
      };
      socket.onerror = () => socket?.close();
    };

    connect();
    return () => {
      closed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, [path]);

  return connected;
}
