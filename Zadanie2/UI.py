def run_ui():
    print("Witaj użytkowniku! \n"
          "Chcesz zrobić? \n"
          "1. Stworzyć sieć \n"
          "2. Nauczyć sieć \n"
          "3. Przetestować sieć \n"
          "4. Wyjść")
    try:
        operation = int(input("->"))
    except ValueError:
        print("Podaj numer opcji")
    try:
        match operation:
            case 1:
                print("Towrzymy wiec siec")
            case 2:
                print("Uczymy wiec siec")
            case 3:
                print("Testujemy wiec siec")
            case 4:
                print("Do zobaczenia!!!")
                exit()
            case _:
                print("Cos zle podales")
                run_ui()
    except UnboundLocalError:
        print("Podana wartość nie jest cyfrą")