package ns

object MemorySpec extends App {

  for (i <- 0 to 100) {
    // println(i)
    ns.zeros(1000, 1000)
    Thread.sleep(10)
  }

  MemoryManager.shutdown()

}
