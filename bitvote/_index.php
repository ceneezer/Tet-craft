<?php
require_once 'inc.head.php';
require_once 'page.login.php';
//posts (tag, search)

if (isset($_POST['issue']) && isset($_POST['vote'])) {
   $vote=intval($_POST['vote']);
   if ($vote>1 && $vote<4) {
      $issue=intval($_POST['issue']);
      $sql='INSERT IGNORE INTO issue_votes (ID_issue, ID_user, ID_vote) values ("'
         .$issue.'", "'.$_SESSION['UID'].'", "'.$vote.'")';
      SQL2ARRAY($sql);
MY_LOG('vote '.$issue.' '.$vote);
   }
}

if (isset($PAGE)) {
   switch($PAGE) {
      case 'Profile':
//not needed for now
?>
<script>
document.body.style.cursor='wait';
alert("Calculating score, this will take a min...");
document.location.href="?page=profile&id=<?=(isset($_REQUEST['id']))?$_REQUEST['id']:'0'?>";
</script>
<?php
         break;
      case 'profile':
         require 'page.profile.php';
         break;
      case 'log':
         require 'page.log.php';
         break;
      case 'me':
?>
<script>
document.body.style.cursor='wait';
alert("Calculating scores, this will take a min...");
document.location.href="?page=mypage";
</script>
<?php
         break;
      case 'mypage':
         require 'page.me.php';
         break;
      case 'issue':
         require 'page.issue.php';
         break;
      case 'issues':
         require 'page.issues.php';
         break;
      case 'tos':
         include 'pub/'.$LANG.'_tos.php';
         break;
      case 'privacy':
         include 'pub/'.$LANG.'_privacy.php';
         break;
      default:
         include 'pub/'.$LANG.'_home.php';
   }
}

require_once 'inc.foot.php';
?>