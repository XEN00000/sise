import argparse
import sys
from utils import (
    load_dataset, save_network, load_network,
    init_error_log, append_error_log, prepare_dataset
)
from network import Network


def run_train(args):
    # 1. Wczytaj surowe
    raw_inputs, raw_labels = load_dataset(
        args.dataset,
        delimiter=args.delimiter,
        skip_header=False
    )

    # 2. Przygotuj (normalizacja + one-hot)
    train_data = prepare_dataset(raw_inputs, raw_labels, n_classes=args.output_size)

    # 3. Inicjalizacja logu
    init_error_log(args.log_path, header="epoch,error")

    # 4. Stwórz sieć
    net = Network(
        layers_config=[args.input_size] + args.hidden_layers + [args.output_size],
        use_bias=not args.no_bias,
        learning_rate=args.learning_rate,
        momentum=args.momentum
    )

    # 5. Trening
    for epoch in range(1, args.max_epochs + 1):
        err = net.train_epoch(train_data, shuffle=not args.no_shuffle)
        if epoch % args.log_every == 0:
            append_error_log(args.log_path, epoch, err)
            print(f"[Epoch {epoch}] error={err:.6f}")
        if args.target_error is not None and err <= args.target_error:
            print(f"Target error reached ({err:.6f} ≤ {args.target_error}) at epoch {epoch}")
            break

    # 6. Zapis sieci
    save_network(net, args.save_path)


def run_test(args):
    # 1. Wczytaj surowe dane i przygotuj je tak samo jak w train
    raw_inputs, raw_labels = load_dataset(
        args.dataset,
        delimiter=args.delimiter,
        skip_header=False
    )
    test_data = prepare_dataset(raw_inputs, raw_labels, n_classes=args.output_size)

    # 2. Wczytaj sieć
    net = load_network(args.network_path)

    # 3. Testowanie
    net.test(
        test_data,
        output_file=args.output_file,
        record_details=args.record_details
    )
    print(f"Test complete. Wyniki zapisane w {args.output_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="MLP: tryb 'train' lub 'test' dla sieci neuronowej typu MLP"
    )
    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- Train ----
    p_train = sub.add_parser("train", help="Trenuj sieć MLP")
    p_train.add_argument("--dataset", required=True, help="Ścieżka do pliku z danymi CSV")
    p_train.add_argument("--input-size", type=int, required=True)
    p_train.add_argument("--output-size", type=int, required=True)
    p_train.add_argument(
        "--hidden-layers", type=int, nargs="+", required=True,
        help="Liczba neuronów w każdej warstwie ukrytej, np. --hidden-layers 5 3"
    )
    p_train.add_argument("--no-bias", action="store_true",
                         help="Wyłącz bias w neuronach przetwarzających")
    p_train.add_argument("--learning-rate", type=float, default=0.1)
    p_train.add_argument("--momentum", type=float, default=0.0)
    p_train.add_argument("--max-epochs", type=int, default=1000)
    p_train.add_argument("--target-error", type=float,
                         help="Przerwij trening, gdy błąd globalny ≤ target-error")
    p_train.add_argument("--log-every", type=int, default=10,
                         help="Co ile epok logować błąd (domyślnie co 10)")
    p_train.add_argument("--log-path", required=True,
                         help="Plik do zapisu logu błędu (CSV)")
    p_train.add_argument("--save-path", required=True,
                         help="Plik do zapisania wytrenowanej sieci (pickle)")
    p_train.add_argument("--delimiter", default=",",
                         help="Separator kolumn w CSV")
    p_train.add_argument("--no-shuffle", action="store_true",
                         help="Nie mieszaj kolejności wzorców każdej epoki")

    # ---- Test ----
    p_test = sub.add_parser("test", help="Testuj wytrenowaną sieć MLP")
    p_test.add_argument("--dataset", required=True, help="Ścieżka do pliku z danymi CSV")
    p_test.add_argument("--input-size", type=int, required=True)
    p_test.add_argument("--output-size", type=int, required=True)
    p_test.add_argument("--network-path", required=True,
                        help="Plik z wytrenowaną siecią (pickle)")
    p_test.add_argument("--output-file", required=True,
                        help="Plik do zapisu wyników testu")
    p_test.add_argument("--record-details", action="store_true",
                        help="Zapisz dodatkowe szczegóły: wagi, wyjścia warstw itd.")
    p_test.add_argument("--delimiter", default=",",
                        help="Separator kolumn w CSV")

    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "test":
        run_test(args)
    else:
        print("Nieznany tryb. Użyj 'train' lub 'test'.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
