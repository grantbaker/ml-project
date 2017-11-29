# Download metadata from IMDB and add it to the csv
import csv
import imdb

ia = imdb.IMDb()


csvfile = 'data/MovieGenre.csv'
newcsv  = 'data/MetaData.csv'

with open(csvfile) as f:
  with open(newcsv, 'w') as new:
    csv_reader = csv.reader(f, delimiter=',')
    skip = True
    for row in csv_reader:
      if skip:
        skip = False
        continue

      actors = []
      director = ""

      try:
        movie_id = row[0]
        movie_object = ia.get_movie(movie_id)
        actors = [ actor.get('name') for actor in movie_object['actors'][:5] ]
        director = movie_object['director'][0].get('name')

      except Exception as e:
        pass

      out_str = ""
      for item in row:
        out_str += item + ","
      out_str = out_str[:-1]
      for actor in actors:
        out_str += actor + "|"
      out_str = out_str[:-1]
      out_str += "," + director + "\n"
      
      print(out_str)
      new.write(out_str)
    
