import { useEffect, useRef, useState } from 'react';

/**
 * Hook to monitor beta concentration and send browser notifications
 * when it drops below a threshold
 */
export function useConcentrationNotifications(latest, threshold = 0) {
  const lastNotificationTime = useRef(0);
  const [permissionStatus, setPermissionStatus] = useState('checking');
  const [lastAlertTime, setLastAlertTime] = useState(null);

  // Request notification permission on mount
  useEffect(() => {
    if ('Notification' in window) {
      if (Notification.permission === 'granted') {
        setPermissionStatus('granted');
      } else if (Notification.permission === 'denied') {
        setPermissionStatus('denied');
      } else {
        setPermissionStatus('requesting');
        Notification.requestPermission().then((permission) => {
          setPermissionStatus(permission);
        });
      }
    } else {
      setPermissionStatus('not-supported');
    }
  }, []);

  // Monitor beta concentration
  useEffect(() => {
    if (!latest || !latest.metrics) {
      return;
    }

    const betaConcentration = latest.metrics.beta_concentration;

    // Only notify if:
    // 1. Beta concentration is below threshold
    // 2. We have permission
    // 3. At least 30 seconds have passed since last notification (to avoid spam)
    const now = Date.now();
    const timeSinceLastNotification = now - lastNotificationTime.current;

    if (
      betaConcentration < threshold &&
      permissionStatus === 'granted' &&
      timeSinceLastNotification > 30000 // 30 seconds
    ) {
      lastNotificationTime.current = now;
      setLastAlertTime(new Date().toLocaleTimeString());

      try {
        const notification = new Notification('Low Concentration Alert! 🧠', {
          body: `Your beta concentration has dropped to ${betaConcentration.toFixed(2)}. Time to refocus!`,
          tag: 'concentration-alert',
          requireInteraction: true,
          silent: false,
        });

        notification.onclick = () => {
          window.focus();
          notification.close();
        };
      } catch (error) {
        console.error('Error creating notification:', error);
      }
    }
  }, [latest, threshold, permissionStatus]);

  return { permissionStatus, lastAlertTime };
}
