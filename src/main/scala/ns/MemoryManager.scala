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
  t.schedule(task, 500L, 1000L)

  def shutdown(): Unit = t.cancel()
}

/*
object THFT extends THFloatTensor with LazyLogging {
  def pointer(t: THFloatTensor): Long = THFloatTensor.getCPtr(t)
  def free(p: Long, t: THFloatTensor): Unit = {
    THJNI.THFloatTensor_free(p, t)
  }
}
*/

/*
object MemoryManager extends LazyLogging {
  // val Threshold: Long = 2L * 1024L * 1024L * 1024L // 2 GB
  val Threshold: Long = 2L * 1024L * 1024L
  val FloatSize = 4
  private val hiMemMark = new AtomicLong(0)
  private val hiTensorMark = new AtomicLong(0)

  def dec(size: Long): Long = {
    hiTensorMark.decrementAndGet()
    val m = hiMemMark.get()
    val s = size * FloatSize
    val t: Long = Math.max(m - s, 0L)
    hiMemMark.set(t)
    t
  }

  def inc(size: Long): Long = {
    hiTensorMark.incrementAndGet()
    hiMemMark.addAndGet(size * FloatSize)
  }

  val t = new java.util.Timer()
  val task: TimerTask = new java.util.TimerTask {
    def run(): Unit = {

      logger.debug(s"running: $hiTensorMark tensors in memory")

      val level = hiMemMark.longValue()
      if (level > Threshold) {
        // logger.debug(s"invoking gc ($level/$Threshold)")
        System.gc()
      }

    }
  }
  t.schedule(task, 500L, 500L)
}
 */
