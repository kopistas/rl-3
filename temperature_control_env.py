import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Параметры среды
        self.current_temp = 20.0
        self.target_temp = 22.0
        self.external_temp = 18.0
        self.max_temp = 30.0
        self.min_temp = 15.0
        
        # Параметры физики
        self.heat_loss_rate = 0.2  # потеря температуры
        self.heater_power = 2.0  # мощность обогревателя
        self.heating_inertia = 0.5  # инертность нагрева
        
        # Время
        self.time_step = 0.1
        self.time = 0.0
        self.max_time = 24.0
        
        # Действия и состояния
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([self.min_temp, self.min_temp, -20.0]),
            high=np.array([self.max_temp, self.max_temp, 40.0]),
            dtype=np.float32
        )
        
        # Дополнительные метрики
        self.heating_cost = 0
        self.total_reward = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_temp = np.random.uniform(18.0, 22.0)
        self.target_temp = 22.0
        self.external_temp = 18.0
        self.time = 0.0
        self.heating_cost = 0
        self.total_reward = 0
        return np.array([self.current_temp, self.target_temp, self.external_temp], dtype=np.float32)

    def step(self, action):
        # Обновляем температуру снаружи
        self.external_temp = 18.0 + 5 * np.sin(2 * np.pi * self.time / 24.0)
        
        # Расчет изменения температуры с учетом инерции
        if action == 1:
            # При включении обогревателя учитываем инерцию
            temp_change = self.heater_power * self.time_step * (1 - self.heating_inertia)
            self.current_temp += temp_change
            self.heating_cost += 1
        
        # Потеря тепла всегда происходит
        heat_loss = self.heat_loss_rate * (self.external_temp - self.current_temp) * self.time_step
        self.current_temp += heat_loss
        
        # Обновляем время
        self.time += self.time_step
        
        # Ограничиваем температуру
        self.current_temp = np.clip(self.current_temp, self.min_temp, self.max_temp)
        
        # Расчет награды
        temp_diff = abs(self.current_temp - self.target_temp)
        base_reward = -temp_diff
        heater_penalty = -0.1 if action == 1 else 0
        reward = base_reward + heater_penalty
        
        # Обновляем общую награду
        self.total_reward += reward
        
        # Условие завершения
        done = (self.time >= self.max_time or 
                self.current_temp <= self.min_temp or 
                self.current_temp >= self.max_temp)
        
        state = np.array([self.current_temp, self.target_temp, self.external_temp], dtype=np.float32)
        return state, reward, done, {
            "total_reward": self.total_reward,
            "heating_cost": self.heating_cost
        }

    def render(self, mode="human"):
        heater_status = "ON" if self.current_temp > self.external_temp else "OFF"
        print(f"Time: {self.time:.2f}h | Current Temp: {self.current_temp:.2f}°C | External Temp: {self.external_temp:.2f}°C | Target Temp: {self.target_temp}°C | Heater: {heater_status} | Heating Cost: {self.heating_cost}")