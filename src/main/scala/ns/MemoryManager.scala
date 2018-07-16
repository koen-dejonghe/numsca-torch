package ns

import java.util.TimerTask
import java.util.concurrent.atomic.AtomicLong

import com.typesafe.scalalogging.LazyLogging

object MemoryManager extends LazyLogging {
  private val tensorCount = new AtomicLong(0)

  def dec: Long = tensorCount.decrementAndGet()
  def inc: Long = tensorCount.incrementAndGet()

  val t = new java.util.Timer()
  val task: TimerTask = new java.util.TimerTask {
    def run(): Unit = {
//      logger.debug(s"running: $tensorCount tensors in memory")
      System.gc()
    }
  }
  t.schedule(task, 250, 250)

  def shutdown(): Unit = t.cancel()
}
