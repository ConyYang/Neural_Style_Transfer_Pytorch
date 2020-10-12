import pygame

# initialize the module
pygame.init()
size = input('Enter the size of your profile photo')
size = int(size)

# create the drawing board
screen = pygame.display.set_mode((size, size))

# load photo
flag = pygame.image.load('flag.png')
photo = pygame.image.load('photo.jpg')

# transform the size of flag
flag = pygame.transform.smoothscale(flag, (size, size))

# paste the flag
screen.blit(photo, (0,0))
screen.blit(flag, (0,0))

# refresh the photo
pygame.display.update()

# save the photo
pygame.image.save(screen, 'new.png')
