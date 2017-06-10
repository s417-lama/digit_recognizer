CC = gcc
CFLAGS = -O3
LDLIBS = -lm
TARGET = ./bin/nnet
OBJDIR = ./obj
OBJS = $(addprefix $(OBJDIR)/, main.o nnet.o optimize.o)
SRCDIR = ./src

$(TARGET):	$(OBJS)
	$(CC) -o $@ $(OBJS) $(LDLIBS)

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	-mkdir -p $(OBJDIR)
	$(CC) $(CFLAGS) -o $@ -c $<

clean:
	rm -f $(OBJS) $(TARGET)
