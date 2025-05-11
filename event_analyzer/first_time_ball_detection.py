def detectar_toques_de_primeira(possession_per_frame, max_frames=16):
    eventos = []
    ultima_posse = None
    inicio = None

    for i, jogador_id in enumerate(possession_per_frame):
        if jogador_id != ultima_posse:
            if ultima_posse is not None:
                fim = i - 1
                duracao = fim - inicio + 1

                if duracao <= max_frames:
                    proximo_jogador = jogador_id
                    if proximo_jogador is not None and proximo_jogador != ultima_posse:
                        eventos.append({
                            "jogador_id": ultima_posse,
                            "start": inicio,
                            "end": fim,
                            "proximo_jogador": proximo_jogador
                        })

            inicio = i
            ultima_posse = jogador_id

    return eventos
