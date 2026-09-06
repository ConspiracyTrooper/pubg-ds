"""Добыча ranked match id и выкачка матчей в data/raw/live/.

Основной путь — «сноуболл» от seed-игроков (см. params.yaml): их истории ->
их ranked-матчи -> все участники этих матчей -> истории участников. Даёт окно
вокруг seed-игроков, а не смещённый в топ срез лидерборда.

Fallback (seed не задан): текущий сезон -> ranked-лидерборд -> истории топ-500.
Ranked-матчи не отдаются через /samples, поэтому только через истории игроков."""

import json
import random

from pubg_ds import config
from pubg_ds.collect.client import PubgClient


# Id текущего сезона
def current_season_id(client: PubgClient, shard: str) -> str:
    data = client.get(f"/shards/{shard}/seasons")
    for season in data["data"]:
        if season["attributes"]["isCurrentSeason"]:
            return season["id"]
    raise RuntimeError("текущий сезон не найден в /seasons")


# Id игроков из ranked-лидерборда (до 500)
def leaderboard_player_ids(client: PubgClient, shard: str, season_id: str, game_mode: str) -> list[str]:
    data = client.get(f"/shards/{shard}/leaderboards/{season_id}/{game_mode}")
    return [p["id"] for p in data.get("included", []) if p["type"] == "player"]


# Account id по точным никам
def player_ids_by_names(client: PubgClient, shard: str, names: list[str]) -> list[str]:
    data = client.get(
        f"/shards/{shard}/players",
        params={"filter[playerNames]": ",".join(names)},
        use_cache=False,
    )
    return [p["id"] for p in data["data"]]


# Account id всех участников уже сохранённых live-матчей (для сноуболла)
def participants_from_saved() -> list[str]:
    ids = []
    for path in sorted(config.DATA_LIVE.glob("*.json")):
        match = json.loads(path.read_text())
        for inc in match.get("included", []):
            if inc["type"] == "participant":
                pid = inc["attributes"]["stats"].get("playerId", "")
                if pid.startswith("account."):
                    ids.append(pid)
    return list(dict.fromkeys(ids))


# Id недавних матчей игроков (все режимы; ranked отфильтруем при выкачке)
def player_match_ids(client: PubgClient, shard: str, player_ids: list[str]) -> list[str]:
    match_ids = []
    # /players принимает до 10 id за запрос
    for i in range(0, len(player_ids), 10):
        chunk = player_ids[i : i + 10]
        data  = client.get(
            f"/shards/{shard}/players",
            params={"filter[playerIds]": ",".join(chunk)},
        )
        for player in data["data"]:
            match_ids += [m["id"] for m in player["relationships"]["matches"]["data"]]
    # Дедуп с сохранением порядка: истории топ-игроков сильно пересекаются
    return list(dict.fromkeys(match_ids))


# Скачиваем матчи, сохраняем только ranked нужного режима
def download_ranked_matches(
    client:    PubgClient,
    shard:     str,
    match_ids: list[str],
    game_mode: str,
    n_target:  int,
) -> int:
    config.DATA_LIVE.mkdir(parents=True, exist_ok=True)
    saved = len(list(config.DATA_LIVE.glob("*.json")))

    for match_id in match_ids:
        if saved >= n_target:
            break
        out = config.DATA_LIVE / f"{match_id}.json"
        if out.exists():
            continue
        data  = client.get(f"/shards/{shard}/matches/{match_id}", use_cache=False)
        attrs = data["data"]["attributes"]
        # Оставляем только ranked нужного режима
        if attrs.get("matchType") != "competitive" or attrs.get("gameMode") != game_mode:
            continue
        out.write_text(json.dumps(data))
        saved += 1

    return saved


# Окно от seed-игроков
def collect_snowball(client: PubgClient, params: dict) -> int:
    shard     = params["shard"]
    game_mode = params["game_mode"]
    n_target  = params["n_matches"]

    seed_ids  = player_ids_by_names(client, shard, params["seed_player_names"])
    seed_hist = player_match_ids(client, shard, seed_ids)
    print(f"seed: {len(seed_ids)} игроков, {len(seed_hist)} матчей в историях")
    saved = download_ranked_matches(client, shard, seed_hist, game_mode, n_target)
    print(f"после seed-историй сохранено: {saved}")
    if saved >= n_target:
        return saved

    # Сноуболл: участники сохранённых матчей -> их истории
    snowball = participants_from_saved()
    random.Random(42).shuffle(snowball)
    snowball = snowball[: params["max_snowball_players"]]
    print(f"сноуболл: {len(snowball)} игроков")

    candidates = player_match_ids(client, shard, snowball)
    print(f"кандидатов в live-окно: {len(candidates)} уникальных матчей")
    return download_ranked_matches(client, shard, candidates, game_mode, n_target)


# Fallback-окно: истории топ-500 игроков(смещено в высокий MMR)
def collect_leaderboard(client: PubgClient, params: dict) -> int:
    shard      = params["shard"]
    season_id  = current_season_id(client, shard)
    player_ids = leaderboard_player_ids(
        client, params["leaderboard_shard"], season_id, params["game_mode"]
    )
    print(f"сезон {season_id}: {len(player_ids)} игроков в лидерборде")

    match_ids = player_match_ids(client, shard, player_ids)
    print(f"кандидатов в live-окно: {len(match_ids)} уникальных матчей")
    return download_ranked_matches(
        client, shard, match_ids, params["game_mode"], params["n_matches"]
    )


def main() -> None:
    params = config.load_params()["collect"]

    with PubgClient() as client:
        # Есть seed-ники -> окно вокруг своего MMR, иначе fallback на лидерборд
        if params.get("seed_player_names"):
            saved = collect_snowball(client, params)
        else:
            saved = collect_leaderboard(client, params)

    print(f"сохранено ranked-матчей: {saved}")


if __name__ == "__main__":
    main()
