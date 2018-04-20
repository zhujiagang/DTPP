#File = open("hello", "w")

#for Index in range(10):
 #   File.write(str(Index) + "\n")

#File.close()

def save(filename, contents):
  fh = open(filename, 'w')
  fh.write(contents)
  fh.close()
save('file.name', 'some tuff')