docker run -it --cpus 10 --rm --runtime=nvidia --name wissem -v /store/fin_news/financial_data:/opt/data -v /net/five/home/wchabchoub/notebook:/opt/notebook -p 11119:8888 financial_ts
