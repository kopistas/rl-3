import pytest
import numpy as np
from temperature_control_env import TemperatureControlEnv

@pytest.fixture
def env():
    return TemperatureControlEnv()

def test_environment_dynamics_up(env):
    initial_temp = env.current_temp
    for _ in range(10): 
        state, _, _, _ = env.step(1)
    assert state[0] > initial_temp, "Температура должна увеличиваться при включении обогревателя"

def test_environment_dynamics_down(env):
    # Сначала нагреваем
    for _ in range(10):
        env.step(1)
    temp_after_heating = env.current_temp
    # Затем охлаждаем
    for _ in range(10):
        state, _, _, _ = env.step(0)
    assert state[0] < temp_after_heating, "Температура должна уменьшаться при выключении обогревателя"

def test_environment_bounds(env):
    env.reset()
    for _ in range(100):  
        state, _, _, _ = env.step(env.action_space.sample())
        assert env.min_temp <= state[0] <= env.max_temp, "Температура должна оставаться в допустимых пределах"

def test_episode_time_limit(env):
    steps = int(env.max_time / env.time_step)
    for i in range(steps + 1):
        _, _, done, _ = env.step(0)
        if i == steps:
            assert done, "Эпизод должен завершиться после 24 часов"

def test_environment_cooling(env):
    env.reset()
    # Сначала нагреваем помещение
    for _ in range(10):
        env.step(1)
    initial_temp = env.current_temp
    # Затем проверяем охлаждение
    for _ in range(20):
        state, _, _, _ = env.step(0)
    assert state[0] < initial_temp, "Температура должна уменьшаться при выключенном обогревателе"

def test_heating_inertia(env):
    env.reset()
    initial_temp = env.current_temp
    # Один шаг нагрева
    _, reward1, _, _ = env.step(1)
    temp_after_first_step = env.current_temp
    # Один шаг остывания
    _, reward2, _, _ = env.step(0)
    temp_after_second_step = env.current_temp
    
    assert temp_after_first_step > initial_temp, "Температура должна увеличиваться после включения обогревателя"
    assert temp_after_second_step < temp_after_first_step, "Температура должна уменьшаться после выключения обогревателя"

def test_heater_penalty(env):
    env.reset()
    initial_reward = env.total_reward
    _, reward, _, _ = env.step(1)
    env.total_reward = initial_reward + reward 
    assert reward < 0, "Должен быть штраф за использование обогревателя"
    assert env.total_reward < initial_reward, "Общая награда должна уменьшаться из-за штрафа за использование обогревателя"

def test_penalty_accumulation(env):
    env.reset()
    initial_cost = env.heating_cost
    for _ in range(10):
        env.step(1)
    assert env.heating_cost == initial_cost + 10, "Штраф должен корректно накапливаться с учётом включений обогревателя"

def test_cooling_without_heater(env):
    env.reset()
    env.current_temp = 25.0
    initial_temp = env.current_temp
    for _ in range(20):
        state, _, _, _ = env.step(0)
    assert state[0] < initial_temp, "Температура должна уменьшаться при выключенном обогревателе"