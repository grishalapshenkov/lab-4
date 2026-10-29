"""
Лабораторная работа № 4. Вариант 14.
Последовательный, двоичный и интерполяционный поиск.

Распределение ключей: А (арифметическая прогрессия A[i] = c + d*i).
Размеры массива: n = 5·10^3, 5·10^4, 5·10^5.
Запросов k = 2000, смешанные: 50 % успешных и 50 % неуспешных.
Дополнительное задание Б7: гистограммы числа сравнений по всем запросам.
"""
import gc
import json
import math
import platform
import random
import statistics
import time

C0, D = 13, 7                     # параметры прогрессии A[i] = C0 + D*i
SIZES = [5_000, 50_000, 500_000]
K = 2000                          # число запросов (1000 успешных + 1000 неуспешных)
REPEATS = 3                       # число повторов замера (берётся медиана)


# ---------------------------------------------------------------- алгоритмы
def linear_search(a, q):
    """Последовательный поиск. Возвращает (индекс, число сравнений)."""
    for i, value in enumerate(a):
        if value == q:
            return i, i + 1
    return -1, len(a)


def binary_search(a, q):
    """Двоичный поиск в упорядоченном массиве (одно сравнение a[mid] == q на шаг)."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


def interpolation_search(a, q):
    """Интерполяционный поиск в упорядоченном массиве."""
    lo, hi, cmps = 0, len(a) - 1, 0
    while lo <= hi and a[lo] <= q <= a[hi]:
        if a[lo] == a[hi]:
            cmps += 1
            return (lo, cmps) if a[lo] == q else (-1, cmps)
        mid = lo + (q - a[lo]) * (hi - lo) // (a[hi] - a[lo])
        cmps += 1
        if a[mid] == q:
            return mid, cmps
        if a[mid] < q:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1, cmps


ALGORITHMS = {"Последовательный": linear_search,
              "Двоичный": binary_search,
              "Интерполяционный": interpolation_search}


# ------------------------------------------------------------------- данные
def make_data(n):
    """Арифметическая прогрессия из n элементов: A[i] = C0 + D*i."""
    return [C0 + D * i for i in range(n)]


def make_queries(a, k, rng):
    """k запросов: половина успешных, половина неуспешных.
    Неуспешный ключ лежит строго между двумя соседними элементами массива,
    поэтому он заведомо отсутствует, но попадает в диапазон [A[0]; A[n-1]]."""
    n = len(a)
    hits = [a[rng.randrange(n)] for _ in range(k // 2)]
    misses = [a[rng.randrange(n - 1)] + rng.randint(1, D - 1) for _ in range(k // 2)]
    queries = [(q, True) for q in hits] + [(q, False) for q in misses]
    rng.shuffle(queries)
    return queries


# ----------------------------------------------------------------- замеры
def benchmark(search_func, data, keys):
    """Среднее время одного запроса, с, и среднее число сравнений."""
    gc.collect()
    start = time.perf_counter()
    total_cmps = 0
    for q in keys:
        idx, cmps = search_func(data, q)
        total_cmps += cmps
    elapsed = time.perf_counter() - start
    return elapsed / len(keys), total_cmps / len(keys)


def collect_comparisons(search_func, data, queries):
    """Число сравнений по каждому запросу (для гистограмм) + проверка корректности."""
    result = []
    for q, present in queries:
        idx, cmps = search_func(data, q)
        assert (idx != -1) == present, "поиск вернул неверный результат"
        if present:
            assert data[idx] == q
        result.append(cmps)
    return result


# --------------------------------------------------------------- теория
def binary_expected(n):
    """Точные средние числа сравнений двоичного поиска (одно сравнение на шаг):
    успешный и неуспешный поиск при равновероятных исходах.
    Дерево поиска сбалансировано: все уровни, кроме последнего, заполнены."""
    h = n.bit_length()                                   # floor(log2 n) + 1
    internal_path = sum(j * 2 ** j for j in range(h - 1)) + (h - 1) * (n - (2 ** (h - 1) - 1))
    succ = (internal_path + n) / n
    unsucc = (internal_path + 2 * n) / (n + 1)
    return succ, unsucc


def theory(n):
    succ_bin, unsucc_bin = binary_expected(n)
    return {
        "lin_hit": (n + 1) / 2, "lin_miss": n, "lin_mix": (3 * n + 1) / 4,
        "bin_log2": math.log2(n), "bin_hit": succ_bin, "bin_miss": unsucc_bin,
        "bin_mix": (succ_bin + unsucc_bin) / 2,
        "int_loglog": math.log2(math.log2(n)),
    }


def self_test():
    """Проверка формулы binary_expected перебором всех ключей малого массива."""
    for n in (1, 2, 7, 10, 100, 1000):
        a = list(range(0, 2 * n, 2))
        hit = sum(binary_search(a, x)[1] for x in a) / n
        miss = sum(binary_search(a, x)[1] for x in range(-1, 2 * n, 2)) / (n + 1)
        s, u = binary_expected(n)
        assert abs(hit - s) < 1e-9 and abs(miss - u) < 1e-9, (n, hit, s, miss, u)


# ----------------------------------------------------------------- график
def make_plots(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(ALGORITHMS)
    ns = [r["n"] for r in res]
    markers = {"Последовательный": "o", "Двоичный": "s", "Интерполяционный": "^"}

    # Рис. 1. Среднее время запроса
    plt.figure(figsize=(6.4, 4.2))
    for name in names:
        plt.plot(ns, [r["time"][name] * 1e6 for r in res], marker=markers[name], label=name)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("n (логарифмическая шкала)"); plt.ylabel("среднее время запроса, мкс")
    plt.title("Среднее время одного запроса"); plt.grid(True, which="both", alpha=.3)
    plt.legend(); plt.tight_layout(); plt.savefig("img/lr4_time.png", dpi=170); plt.close()

    # Рис. 2. Среднее число сравнений
    plt.figure(figsize=(6.4, 4.2))
    for name in names:
        plt.plot(ns, [r["cmps"][name] for r in res], marker=markers[name], label=name + " (опыт)")
    plt.plot(ns, [r["theory"]["lin_mix"] for r in res], "k--", lw=1, label=r"$(3n+1)/4$")
    plt.plot(ns, [r["theory"]["bin_log2"] for r in res], "k:", lw=1.2, label=r"$\log_2 n$")
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel("n (логарифмическая шкала)"); plt.ylabel("среднее число сравнений")
    plt.title("Среднее число сравнений на запрос"); plt.grid(True, which="both", alpha=.3)
    plt.legend(fontsize=8); plt.tight_layout(); plt.savefig("img/lr4_cmps.png", dpi=170); plt.close()

    # Рис. 3. Гистограммы (строки — алгоритмы, столбцы — n)
    fig, axes = plt.subplots(3, len(res), figsize=(11, 8.2))
    for j, r in enumerate(res):
        for i, name in enumerate(names):
            ax = axes[i][j]
            data = r["all_cmps"][name]
            mean = statistics.mean(data)
            if name == "Последовательный":
                ax.hist(data, bins=40, color="#4c72b0")
                ax.set_yscale("log")
                ax.axvline(r["theory"]["lin_mix"], color="k", ls="--", lw=1.2,
                           label=f"теория (3n+1)/4 = {r['theory']['lin_mix']:.0f}")
            elif name == "Двоичный":
                lo, hi = min(data), max(data)
                ax.hist(data, bins=[x - .5 for x in range(lo, hi + 2)], color="#55a868")
                ax.axvline(r["theory"]["bin_log2"], color="k", ls=":", lw=1.4,
                           label=f"log₂n = {r['theory']['bin_log2']:.1f}")
            else:
                ax.hist(data, bins=[x - .5 for x in range(min(data), max(data) + 2)], color="#c44e52")
                ax.set_xlim(-1, 6)
            ax.axvline(mean, color="red", lw=1.4, label=f"опыт, среднее = {mean:.1f}")
            ax.set_title(f"{name}, n = {r['n']:,}".replace(",", " "), fontsize=9)
            ax.legend(fontsize=7)
            if j == 0:
                ax.set_ylabel("число запросов")
            if i == 2:
                ax.set_xlabel("число сравнений")
    plt.tight_layout(); plt.savefig("img/lr4_hist.png", dpi=150); plt.close()


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    self_test()
    rng = random.Random(14)
    results = []
    print(f"Вариант 14: A[i] = {C0} + {D}*i, k = {K} (50/50), повторов {REPEATS}")
    for n in SIZES:
        data = make_data(n)
        queries = make_queries(data, K, rng)
        keys = [q for q, _ in queries]
        row = {"n": n, "time": {}, "cmps": {}, "all_cmps": {}, "stat": {}, "theory": theory(n)}
        print(f"\nn = {n}")
        for name, func in ALGORITHMS.items():
            runs = [benchmark(func, data, keys) for _ in range(REPEATS)]
            t = statistics.median(x[0] for x in runs)
            c = runs[0][1]
            per_query = collect_comparisons(func, data, queries)
            hit = [x for x, (_, p) in zip(per_query, queries) if p]
            miss = [x for x, (_, p) in zip(per_query, queries) if not p]
            row["time"][name], row["cmps"][name] = t, c
            row["all_cmps"][name] = per_query
            row["stat"][name] = {"mean": statistics.mean(per_query),
                                 "hit": statistics.mean(hit), "miss": statistics.mean(miss),
                                 "min": min(per_query), "max": max(per_query)}
            print(f"  {name:<18} {t * 1e6:>11.2f} мкс/запрос {c:>12.2f} сравнений")
        results.append(row)

    with open("res/lr4.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    make_plots(results)
