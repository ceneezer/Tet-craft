<?php
//https://192.168.79.4/mine/voteprism/?page=issue&id=3989
//https://192.168.79.4/mine/voteprism/?page=issue&id=3988
/*
2668
3581
3984
3985
3986
3987
*/
   $sql='SELECT ssid, mpid, ';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='title as title, DATE_FORMAT(deadline, "%Y-%m-%e") AS d FROM issues WHERE id="'.SQLIFY($ID).'"';
   $sql=SQL2ARRAY($sql);   
   if (count($sql)>0) {
      $sql=$sql[0];
      if ($sql['ssid']>0) {
         $link='https://sencanada.ca/'.(($LANG=='fr')?'fr/dans-la-chambre':'en/in-the-chamber').'/votes/details/'.$sql['ssid'].'/';
      } else {
         $link=$sql['mpid'];
         if ($LANG=='fr') {
            $link=str_replace('/en/', '/fr/', $link);
            $link=str_replace('.ourcommons.', '.noscommunes.', $link);
         }
      }
?>
   <main>
      <h3>
<a href="<?=$link?>" target="_blank"><?=$sql['d']?> <?=$sql['title']?></a>
      </h3>
<?php
   $sql='SELECT COUNT(v.id) as num, SUM(v.Vvalue) as s FROM issue_votes AS i LEFT JOIN votes AS v ON v.id=i.ID_vote WHERE i.ID_issue="'.SQLIFY($ID).'"';
   $sql=SQL2ARRAY($sql);
   if (count($sql)>0) {
      $sql=$sql[0];
?>  
<h4><?=SQLATE('site votes', $LANG, true)?>:<?=$sql['num']?>  
(<?=$sql['s']?> <?=SQLATE('for', $LANG, true)?>)</h4>
<?php 
   }
   if (isset($_SESSION['UID'])) { 
      $sql='SELECT v.';
      if ($LANG=='fr')
         $sql.='fr';
      $sql.='caption as caption, i.id, v.frcaption, i.stamp, CONCAT(u.ssid, u.mpid) AS mky FROM issue_votes AS i LEFT JOIN votes AS v on v.id=i.ID_vote LEFT JOIN users AS u ON u.id=i.ID_user WHERE i.ID_issue="'.SQLIFY($ID).'" AND i.ID_user="'.SQLIFY($_SESSION['UID']).'"';
      $sql=SQL2ARRAY($sql);
      if (count($sql)>0) {
         $sql=$sql[0];
         // ideally here we need a piece of private info to cover hash - unchanging through an update, but not published... mky?!  ty ;)
         //... update users set mpid=???? where mpid=''
?>
      <h5><?=SQLATE('your', $LANG, true)?> vote: <?=$sql['caption']?> <?=substr(hash('sha512',$salt[0].$sql['id'].$sql['stamp'].substr('.'.$sql['mky'],-10,4).$sql['frcaption'].$_SESSION['UID'].$salt[1]),-50)?></h5>
<?php } else { ?>
      <form method="post" id="voteYea" class="vote">
         <input type="hidden" name="issue" value="<?=$ID?>" />
         <input type="hidden" name="vote" value="2" />
         <input type="submit" value="<?=SQLATE('yea', $LANG, true)?>"  onmo.useup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" />
      </form>
      <form method="post" id="voteNay" class="vote">
         <input type="hidden" name="issue" value="<?=$ID?>" />
         <input type="hidden" name="vote" value="3" />
         <input type="submit" value="<?=SQLATE('nay', $LANG, true)?>"  onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" />
      </form>
<?php }} ?>
      <section>
         <table id="tblVotes">
            <tr>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 0);"><?=SQLATE('vote', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 1);"><?=SQLATE('name', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 2);"><?=SQLATE('telephone', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 3);"><?=SQLATE('e-mail', $LANG, true)?></th>
               <th onmouseup="this.classList.remove('wait');" onmousedown="this.classList.add('wait');" onclick="sortTable('tblVotes', 4);"><?=SQLATE('party', $LANG, true)?></th>
            </tr>
<?php
   $sql='SELECT u.phone, p.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as party, r.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as region, u.email, u.lname, i.ID_user, v.';
   if ($LANG=='fr')
      $sql.='fr';
   $sql.='caption as caption FROM issue_votes AS i LEFT JOIN votes AS v ON v.id=i.ID_vote LEFT JOIN users AS u ON u.id=i.ID_user LEFT JOIN regions AS r ON r.id=u.ID_region LEFT JOIN parties AS p ON p.id=u.ID_party WHERE u.link IS NOT NULL AND ID_issue="'.SQLIFY($ID).'"';
   if (isset($_REQUEST['search']) && $_REQUEST['search']!='') {
      $search=str_replace('%', '', $search);
      $search=SQLIFY($_REQUEST['search']);
      $sql.=' AND (';
      $sql.='p.caption LIKE "%'.$search.'%"';
      $sql.=' OR p.frcaption LIKE "%'.$search.'%"';
      $sql.=' OR r.caption LIKE "%'.$search.'%"';
      $sql.=' OR r.frcaption LIKE "%'.$search.'%"';
      $sql.=' OR u.fname LIKE "%'.$search.'%"';
      $sql.=' OR u.lname LIKE "%'.$search.'%"';
      $sql.=' OR u.link LIKE "%'.$search.'%"';
      $sql.=' OR u.phone LIKE "%'.$search.'%"';
      $sql.=' OR u.email LIKE "%'.$search.'%"';
      $sql.=' OR u.ssid LIKE "%'.$search.'%"';
      $sql.=' OR u.mpid LIKE "%'.$search.'%"';
      $sql.=')';
   }
   $sql=SQL2ARRAY($sql);
   foreach ($sql as $s) {
?>
            <tr>
               <td><?=$s['caption']?></td>
               <td><a href='?page=profile&id=<?=$s['ID_user']?>'><?=$s['lname']?></a></td>
               <td><?=$s['phone']?></td>
               <td><?=$s['email']?></td>
               <td><?=$s['party']?></td>
            </tr>
<?php } ?>
         </table>
      </section>
   </main>
<?php } ?>