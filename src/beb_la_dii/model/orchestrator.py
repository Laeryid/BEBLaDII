import torch

class Orchestrator:
    """
    Алгоритмический роутер (Оркестратор) для управления токенами в диффузионном ядре.
    Принимает решения исключительно на базе тензорных метрик от Головы уверенности (Confidence Head) и Ансамбля,
    без доступа к строкам словаря или оригинальному тексту.
    """
    
    def __init__(self, delta_threshold=0.05, conflict_sim_threshold=0.30, ignore_sim_threshold=0.70):
        self.delta_threshold = delta_threshold
        self.conflict_sim_threshold = conflict_sim_threshold
        self.ignore_sim_threshold = ignore_sim_threshold

    def analyze_token(self, t_local, raw_d_prox, delta, conflict_sim):
        """
        Анализирует состояние одного токена и возвращает рекомендацию.
        
        Args:
            t_local (float): Текущий уровень локального шума (0.0 - 1.0).
            raw_d_prox (float): Косинусное сходство с ближайшим словом (Top1_Sim).
            delta (float): Разница между Top1_Sim и Top2_Sim.
            conflict_sim (float): Косинусное сходство между векторами Top1 и Top2.
            
        Returns:
            dict: Статус и рекомендуемое действие ('CONTINUE', 'DECODER_READY', 'CLM_HELP').
        """
        if t_local > 0.0:
            return {
                "status": "DENOISING",
                "action": "CONTINUE",
                "reason": f"Token is still denoising (t={t_local:.3f})"
            }
            
        if delta >= self.delta_threshold:
            return {
                "status": "CRYSTALLIZED",
                "action": "DECODER_READY",
                "reason": f"Token crystallized confidently (Delta={delta:.3f})"
            }
            
        # Если Delta < delta_threshold, токен балансирует (ambiguous)
        if conflict_sim >= self.ignore_sim_threshold:
            return {
                "status": "SYNTAX_VARIATION",
                "action": "DECODER_READY",
                "reason": f"Balancing between highly similar tokens (ConfSim={conflict_sim:.3f}). Decoder will resolve."
            }
            
        if conflict_sim <= self.conflict_sim_threshold:
            return {
                "status": "SEMANTIC_CONFLICT",
                "action": "CLM_HELP",
                "reason": f"Balancing between orthogonal tokens (ConfSim={conflict_sim:.3f}). Needs RAG/LLM context."
            }
            
        # Серая зона (между 0.30 и 0.70)
        return {
            "status": "MILD_AMBIGUITY",
            "action": "CLM_HELP",
            "reason": f"Uncertain semantic variation (ConfSim={conflict_sim:.3f}). Safer to ask CLM."
        }

    def batch_analyze(self, t_local_batch, raw_d_prox_batch, delta_batch, conflict_sim_batch):
        """
        Пакетный анализ токенов для всей секвенции.
        """
        if isinstance(t_local_batch, torch.Tensor):
            B, T = t_local_batch.shape
        else:
            B = len(t_local_batch)
            T = len(t_local_batch[0])
            
        actions = []
        for b in range(B):
            seq_actions = []
            for t in range(T):
                t_loc = t_local_batch[b, t].item() if isinstance(t_local_batch, torch.Tensor) else t_local_batch[b][t]
                raw_d = raw_d_prox_batch[b, t].item() if isinstance(raw_d_prox_batch, torch.Tensor) else raw_d_prox_batch[b][t]
                dl = delta_batch[b, t].item() if isinstance(delta_batch, torch.Tensor) else delta_batch[b][t]
                cs = conflict_sim_batch[b, t].item() if isinstance(conflict_sim_batch, torch.Tensor) else conflict_sim_batch[b][t]
                
                res = self.analyze_token(t_loc, raw_d, dl, cs)
                seq_actions.append(res)
            actions.append(seq_actions)
        return actions

if __name__ == "__main__":
    orchestrator = Orchestrator()
    # Тест: Токен "over" vs "Over" (Синтаксический шум)
    res1 = orchestrator.analyze_token(t_local=0.0, raw_d_prox=0.599, delta=0.046, conflict_sim=0.790)
    print("Test 1 (Syntax Variation):", res1)
    
    # Тест: Токен "on" vs "exchange" (Семантический тупик)
    res2 = orchestrator.analyze_token(t_local=0.0, raw_d_prox=0.450, delta=0.012, conflict_sim=0.180)
    print("Test 2 (Semantic Conflict):", res2)
    
    # Тест: Токен идеально закристаллизовался
    res3 = orchestrator.analyze_token(t_local=0.0, raw_d_prox=0.950, delta=0.450, conflict_sim=0.100)
    print("Test 3 (Crystallized):", res3)
