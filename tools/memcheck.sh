#!/bin/sh
compute-sanitizer --tool memcheck --leak-check full --track-stream-ordered-races all $1
