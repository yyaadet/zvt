import click



@click.command()
@click.option("--path", default="./rmd/coal.csv", help="daily stock data")
def main(path):
    pass


if __name__ == "__main__":
    main()
